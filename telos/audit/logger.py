"""
Audit Logger Module

Provides cryptographic audit trails for all governance operations.
Every intervention is logged with a hash chain for tamper detection.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import logging

from telos.interventions.steering_config import Intervention

logger = logging.getLogger("audit_logger")


@dataclass
class AuditRecord:
    """
    A single audit record for a governed inference request.
    
    This record is cryptographically signed to ensure integrity.
    """
    # Identification
    record_id: str
    timestamp: str
    
    # Input/Output
    prompt: str
    response: str
    
    # Governance details
    interventions: List[Dict[str, Any]]
    steering_enabled: bool
    capping_enabled: bool
    assistant_axis_stability: float
    
    # Cryptographic
    previous_hash: str
    current_hash: str
    
    # Metadata
    model: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """Create from dictionary."""
        return cls(**data)


class AuditLogger:
    """
    Manages audit logging for governance operations.
    
    Provides:
    - Cryptographic hash chain for tamper detection
    - JSON file storage
    - Query and retrieval capabilities
    """
    
    def __init__(
        self,
        log_dir: str = "audit_logs",
        session_id: Optional[str] = None,
    ):
        """
        Initialize the audit logger.
        
        Args:
            log_dir: Directory to store audit logs
            session_id: Optional session ID for grouping requests
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = session_id or self._generate_session_id()
        self._previous_hash = "0" * 64  # Genesis hash
        
        # Current session file
        self.session_file = self.log_dir / f"session_{self.session_id}.jsonl"
        
        # Stats
        self.record_count = 0
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return hashlib.sha256(
            datetime.now().isoformat().encode()
        ).hexdigest()[:16]
    
    def _compute_hash(
        self,
        prompt: str,
        response: str,
        interventions: List[Intervention],
        previous_hash: str,
    ) -> str:
        """
        Compute SHA-256 hash for the record.
        
        The hash includes:
        - Previous hash (chain)
        - Prompt
        - Response
        - All intervention details
        """
        # Build content string
        content_parts = [
            previous_hash,
            prompt,
            response,
            str(len(interventions)),
        ]
        
        for intervention in interventions:
            content_parts.extend([
                str(intervention.feature_id),
                str(intervention.layer),
                str(intervention.coefficient),
            ])
        
        content = "|".join(content_parts)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def log(
        self,
        prompt: str,
        response: str,
        interventions: List[Intervention],
        stability: float,
        steering_enabled: bool = True,
        capping_enabled: bool = True,
        model: str = "unknown",
        **metadata,
    ) -> AuditRecord:
        """
        Log a governed inference request.
        
        Args:
            prompt: Input prompt
            response: Generated response
            interventions: List of interventions applied
            stability: Assistant axis stability score
            steering_enabled: Whether steering was enabled
            capping_enabled: Whether capping was enabled
            model: Model name
            **metadata: Additional metadata
            
        Returns:
            Created AuditRecord
        """
        # Compute hash
        current_hash = self._compute_hash(
            prompt=prompt,
            response=response,
            interventions=interventions,
            previous_hash=self._previous_hash,
        )
        
        # Create record
        record = AuditRecord(
            record_id=hashlib.sha256(
                f"{prompt}{current_hash}".encode()
            ).hexdigest()[:16],
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            interventions=[
                i.to_dict() for i in interventions
            ],
            steering_enabled=steering_enabled,
            capping_enabled=capping_enabled,
            assistant_axis_stability=stability,
            previous_hash=self._previous_hash,
            current_hash=current_hash,
            model=model,
            session_id=self.session_id,
        )
        
        # Write to file
        self._write_record(record)
        
        # Update chain
        self._previous_hash = current_hash
        self.record_count += 1
        
        logger.debug(f"Logged audit record: {record.record_id}")
        
        return record
    
    def _write_record(self, record: AuditRecord) -> None:
        """Write record to JSONL file."""
        with open(self.session_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + "\n")
    
    def verify_chain(self) -> Dict[str, Any]:
        """
        Verify the integrity of the hash chain.
        
        Returns:
            Dictionary with verification results
        """
        if not self.session_file.exists():
            return {
                "valid": False,
                "error": "No session file found",
            }
        
        records = []
        with open(self.session_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                records.append(AuditRecord.from_dict(data))
        
        if not records:
            return {
                "valid": True,
                "record_count": 0,
                "message": "Empty chain",
            }
        
        # Verify chain
        prev_hash = "0" * 64
        for record in records:
            if record.previous_hash != prev_hash:
                return {
                    "valid": False,
                    "record_id": record.record_id,
                    "error": f"Chain broken at {record.record_id}",
                }
            
            # Recompute hash
            interventions = [
                Intervention.from_dict(i) 
                for i in record.interventions
            ]
            expected_hash = self._compute_hash(
                prompt=record.prompt,
                response=record.response,
                interventions=interventions,
                previous_hash=record.previous_hash,
            )
            
            if expected_hash != record.current_hash:
                return {
                    "valid": False,
                    "record_id": record.record_id,
                    "error": f"Hash mismatch at {record.record_id}",
                }
            
            prev_hash = record.current_hash
        
        return {
            "valid": True,
            "record_count": len(records),
            "latest_hash": records[-1].current_hash,
        }
    
    def get_records(
        self,
        limit: Optional[int] = None,
    ) -> List[AuditRecord]:
        """Get audit records from the current session."""
        if not self.session_file.exists():
            return []
        
        records = []
        with open(self.session_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                records.append(AuditRecord.from_dict(data))
                
                if limit and len(records) >= limit:
                    break
        
        return records
    
    def export_session(self, output_path: Optional[str] = None) -> str:
        """
        Export the current session to a JSON file.
        
        Args:
            output_path: Optional output path
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = str(
                self.log_dir / f"session_{self.session_id}_export.json"
            )
        
        records = self.get_records()
        
        with open(output_path, 'w') as f:
            json.dump(
                [r.to_dict() for r in records],
                f,
                indent=2,
            )
        
        logger.info(f"Exported {len(records)} records to {output_path}")
        
        return output_path

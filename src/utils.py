import hashlib


def file_checksum(content: str, hash_length: int = 16) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:hash_length]

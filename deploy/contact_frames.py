"""Read a recent complete recording frame while the writer is still active."""
import time
from pathlib import Path
import cv2


def latest_complete_frame(folder, max_age_s=2., now=None):
    now = time.time() if now is None else now
    # The newest PNG may still be being compressed. Older complete frames are
    # eligible only within the same freshness bound; never hide a stalled writer.
    for path in sorted(Path(folder).glob('raw_*.png'), reverse=True)[:4]:
        age = now - path.stat().st_mtime
        if not 0 <= age <= max_age_s:
            continue
        frame = cv2.imread(str(path))
        if frame is not None and frame.size:
            return path, frame
    raise RuntimeError('no recent complete overhead frame; recording may be incomplete or stalled')

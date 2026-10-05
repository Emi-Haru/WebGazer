/**
 * Capture an unmirrored frame from a <video> as a JPEG Blob.
 * Preview may be CSS-mirrored; the bitmap itself is left as the camera sees it
 * (matches training/extract flip_horizontal: false).
 */

/** Same constraints as personal training JPEGs — keeps ipd/face_w/z in-distribution for MLP. */
export const COLLECT_VIDEO_CONSTRAINTS: MediaTrackConstraints = {
  facingMode: 'user',
  width: { ideal: 1280 },
  height: { ideal: 720 }
};

export async function captureVideoFrame (
  video: HTMLVideoElement,
  quality = 0.92
): Promise<Blob> {
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    throw new Error('Camera frame not ready yet — wait a moment and try again.');
  }
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    throw new Error('Camera has no dimensions yet.');
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: false });
  if (!ctx) throw new Error('Could not create canvas context.');
  ctx.drawImage(video, 0, 0, width, height);

  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob((b) => resolve(b), 'image/jpeg', quality);
  });
  if (!blob) throw new Error('Failed to encode JPEG frame.');
  return blob;
}

export async function startCamera (video: HTMLVideoElement): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: COLLECT_VIDEO_CONSTRAINTS
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function stopCamera (stream: MediaStream | null): void {
  stream?.getTracks().forEach((t) => t.stop());
}

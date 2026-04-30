
import React from 'react';

export default function Upload({ file, previewUrl, onFileChange }) {
  return (
    <section className="upload-box">
      <label className="drop-zone">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => onFileChange(e.target.files?.[0] || null)}
        />
        <span className="upload-icon">+</span>
        <span className="upload-title">Upload handwriting image</span>
        <span className="upload-subtitle">
          JPG, PNG, or scanned handwritten text
        </span>
      </label>

      {previewUrl && (
        <div className="preview-panel">
          <img src={previewUrl} alt="Selected handwriting preview" />
          <div>
            <span className="eyebrow">Selected file</span>
            <strong>{file?.name}</strong>
            <small>{Math.max(1, Math.round((file?.size || 0) / 1024))} KB</small>
          </div>
        </div>
      )}
    </section>
  );
}

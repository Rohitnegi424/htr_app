
import React from 'react';

export default function Result({ text, translatedText, targetLabel }) {
  if (!text && !translatedText) return null;

  return (
    <section className="result-grid">
      {text && (
        <article className="result-box">
          <div className="section-heading">
            <span className="eyebrow">OCR result</span>
            <h2>Recognized text</h2>
          </div>
          <p>{text}</p>
        </article>
      )}

      {translatedText && (
        <article className="result-box translated">
          <div className="section-heading">
            <span className="eyebrow">{targetLabel}</span>
            <h2>Translation</h2>
          </div>
          <p>{translatedText}</p>
        </article>
      )}
    </section>
  );
}

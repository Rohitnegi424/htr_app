
import React from 'react';

export default function Controls({
  canUseText,
  isSpeaking,
  languageOptions,
  speechLang,
  speechSource,
  targetLang,
  hasTranslatedText,
  onSpeechLangChange,
  onSpeechSourceChange,
  onTargetLangChange,
  onSpeak,
  onStopSpeaking,
  onTranslate,
  onAutocorrect,
  translating,
  autocorrecting,
}) {
  const targetOptions = languageOptions.filter((language) => language.translateCode);

  return (
    <section className="controls">
      <div className="control-field">
        <label htmlFor="target-language">Translate to</label>
        <select
          id="target-language"
          value={targetLang}
          onChange={(e) => onTargetLangChange(e.target.value)}
          disabled={!canUseText || translating}
        >
          {targetOptions.map((language) => (
            <option key={language.translateCode} value={language.translateCode}>
              {language.label}
            </option>
          ))}
        </select>
      </div>

      <button
        className="secondary"
        type="button"
        onClick={onTranslate}
        disabled={!canUseText || translating}
      >
        {translating ? 'Translating...' : 'Translate'}
      </button>

      <button
        className="secondary"
        type="button"
        onClick={onAutocorrect}
        disabled={!canUseText || autocorrecting}
      >
        {autocorrecting ? 'Correcting...' : 'Autocorrect OCR'}
      </button>

      <div className="control-field">
        <label htmlFor="voice-language">Read aloud language</label>
        <select
          id="voice-language"
          value={speechLang}
          onChange={(e) => onSpeechLangChange(e.target.value)}
          disabled={!canUseText}
        >
          {languageOptions.map((language) => (
            <option key={language.speechCode} value={language.speechCode}>
              {language.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-field">
        <label htmlFor="speech-source">Speak from</label>
        <select
          id="speech-source"
          value={speechSource}
          onChange={(e) => onSpeechSourceChange(e.target.value)}
          disabled={!canUseText}
        >
          <option value="ocr">OCR text</option>
          <option value="translation" disabled={!hasTranslatedText}>
            Translated text
          </option>
        </select>
      </div>

      <button type="button" onClick={onSpeak} disabled={!canUseText}>
        Read aloud
      </button>

      <button
        className="danger"
        type="button"
        onClick={onStopSpeaking}
        disabled={!isSpeaking}
      >
        Stop speaking
      </button>
    </section>
  );
}

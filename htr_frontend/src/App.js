
import React, { useEffect, useMemo, useRef, useState } from 'react';
import Upload from './components/Upload';
import Result from './components/Result';
import Controls from './components/Controls';
import {
  autocorrectText,
  getTrocrStatus,
  sendImage,
  synthesizeSpeech,
  translateText,
} from './services/api';

const languageOptions = [
  { label: 'English', speechCode: 'en-US', translateCode: 'en', ttsCode: 'en' },
  { label: 'Hindi', speechCode: 'hi-IN', translateCode: 'hi', ttsCode: 'hi' },
  { label: 'Spanish', speechCode: 'es-ES', translateCode: 'es', ttsCode: 'es' },
  { label: 'French', speechCode: 'fr-FR', translateCode: 'fr', ttsCode: 'fr' },
  { label: 'German', speechCode: 'de-DE', translateCode: 'de', ttsCode: 'de' },
  { label: 'Tamil', speechCode: 'ta-IN', translateCode: 'ta', ttsCode: 'ta' },
  { label: 'Arabic', speechCode: 'ar-SA', translateCode: 'ar', ttsCode: 'ar' },
  { label: 'Bengali', speechCode: 'bn-IN', translateCode: 'bn', ttsCode: 'bn' },
  { label: 'Gujarati', speechCode: 'gu-IN', translateCode: 'gu', ttsCode: 'gu' },
  { label: 'Kannada', speechCode: 'kn-IN', translateCode: 'kn', ttsCode: 'kn' },
  { label: 'Malayalam', speechCode: 'ml-IN', translateCode: 'ml', ttsCode: 'ml' },
  { label: 'Marathi', speechCode: 'mr-IN', translateCode: 'mr', ttsCode: 'mr' },
  { label: 'Telugu', speechCode: 'te-IN', translateCode: 'te', ttsCode: 'te' },
  { label: 'Urdu', speechCode: 'ur-IN', translateCode: 'ur', ttsCode: 'ur' },
];

const ocrModels = {
  custom: 'Custom HTR',
  hindi: 'Hindi HTR',
  trocr: 'TrOCR',
};

const compactError = (message) => {
  if (!message) return 'Prediction failed. Check backend server and try again.';
  if (message.includes('Unrecognized image processor')) {
    return 'TrOCR model metadata is incomplete. The backend is trying a compatible processor fallback.';
  }
  return message.length > 220 ? `${message.slice(0, 220)}...` : message;
};

export default function App() {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [targetLang, setTargetLang] = useState('hi');
  const [speechLang, setSpeechLang] = useState('en-US');
  const [speechSource, setSpeechSource] = useState('ocr');
  const [history, setHistory] = useState([]);
  const [ocrModel, setOcrModel] = useState('custom');
  const [lastModel, setLastModel] = useState('');
  const [loading, setLoading] = useState(false);
  const [translating, setTranslating] = useState(false);
  const [autocorrecting, setAutocorrecting] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [message, setMessage] = useState('');
  const audioRef = useRef(null);
  const audioUrlRef = useRef('');
  const speechRunRef = useRef(0);

  const previewUrl = useMemo(() => {
    if (!file) return '';
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, []);

  const targetLabel =
    languageOptions.find((language) => language.translateCode === targetLang)?.label ||
    'Selected language';

  const selectedSpeechText = speechSource === 'translation' ? translatedText : text;
  const selectedSpeechLanguage =
    languageOptions.find((language) => language.speechCode === speechLang) ||
    languageOptions[0];

  const formatTrocrStatus = (status) => {
    const percent = Number.isFinite(status?.percent) ? status.percent : 0;
    const messageText = status?.message || 'Preparing TrOCR...';
    return `${messageText} ${percent}%`;
  };

  const handleFileChange = (nextFile) => {
    setFile(nextFile);
    setText('');
    setTranslatedText('');
    setSpeechSource('ocr');
    setLastModel('');
    setMessage(nextFile ? 'Image ready. You can recognize it as many times as you need.' : '');
  };

  const runPrediction = async (modelName) => {
    if (!file) return;
    let statusTimer = null;

    setOcrModel(modelName);
    setLoading(true);
    setMessage(
      modelName === 'trocr'
        ? 'Preparing TrOCR... 0%'
        : modelName === 'hindi'
          ? 'Running Hindi HTR...'
        : ''
    );

    if (modelName === 'trocr') {
      statusTimer = window.setInterval(async () => {
        try {
          const status = await getTrocrStatus();
          setMessage(formatTrocrStatus(status));
        } catch (error) {
          setMessage('Preparing TrOCR...');
        }
      }, 1000);
    }

    try {
      const res = await sendImage(file, modelName, speechLang);
      const recognizedText = res.text || 'No text detected';
      setText(recognizedText);
      setTranslatedText('');
      setSpeechSource('ocr');
      setLastModel(res.modelLabel || ocrModels[modelName] || 'Selected model');
      setMessage(`Recognized with ${res.modelLabel || ocrModels[modelName] || 'selected model'}.`);
      setHistory((items) => [
        {
          id: Date.now(),
          fileName: file.name,
          model: res.modelLabel || ocrModels[modelName] || 'Selected model',
          text: recognizedText,
          time: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        },
        ...items.slice(0, 4),
      ]);
    } catch (error) {
      console.error('Prediction failed', error);
      const backendMessage = error.response?.data?.error;
      setText('');
      setMessage(compactError(backendMessage));
    } finally {
      if (statusTimer) {
        window.clearInterval(statusTimer);
      }
      setLoading(false);
    }
  };

  const handlePredict = () => {
    runPrediction(ocrModel);
  };

  const handleCancel = () => {
    speechRunRef.current += 1;
    audioRef.current?.pause();
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = '';
    }
    setFile(null);
    setText("");
    setTranslatedText('');
    setSpeechSource('ocr');
    setLastModel('');
    setMessage('');
    setIsSpeaking(false);
  };

  const handleTranslate = async () => {
    if (!text) return;
    setTranslating(true);
    setMessage('');

    try {
      const translated = await translateText(text, targetLang);
      setTranslatedText(translated || 'Translation unavailable for this text.');
      const voice = languageOptions.find(
        (language) => language.translateCode === targetLang
      );
      if (voice) {
        setSpeechLang(voice.speechCode);
      }
      setSpeechSource('translation');
    } catch (error) {
      console.error('Translation failed', error);
      setMessage('Translation failed. Check your internet connection and try again.');
    } finally {
      setTranslating(false);
    }
  };

  const handleAutocorrect = async () => {
    if (!text) return;

    setAutocorrecting(true);
    setMessage('Autocorrecting OCR text...');

    try {
      const result = await autocorrectText(text, speechLang);
      setText(result.corrected || text);
      setTranslatedText('');
      setSpeechSource('ocr');
      setMessage(
        result.changed
          ? 'OCR text was autocorrected.'
          : 'No autocorrect changes were needed.'
      );
    } catch (error) {
      console.error('Autocorrect failed', error);
      const backendMessage = error.response?.data?.error;
      setMessage(backendMessage || 'Autocorrect failed. Check backend/internet and try again.');
    } finally {
      setAutocorrecting(false);
    }
  };

  const handleSpeak = () => {
    if (!selectedSpeechText) {
      setMessage(
        speechSource === 'translation'
          ? 'Translate the OCR text first, then read the translated text aloud.'
          : 'Recognize text first, then read it aloud.'
      );
      return;
    }

    const runId = speechRunRef.current + 1;
    speechRunRef.current = runId;
    audioRef.current?.pause();

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = '';
    }

    setIsSpeaking(true);
    setMessage(`Preparing ${selectedSpeechLanguage.label} audio...`);

    synthesizeSpeech(selectedSpeechText, selectedSpeechLanguage.ttsCode)
      .then((audioUrl) => {
        if (speechRunRef.current !== runId) {
          URL.revokeObjectURL(audioUrl);
          return;
        }

        audioUrlRef.current = audioUrl;
        const audio = new Audio(audioUrl);
        audioRef.current = audio;
        audio.onended = () => setIsSpeaking(false);
        audio.onerror = () => {
          setIsSpeaking(false);
          setMessage('Could not play the generated audio.');
        };
        setMessage('');
        return audio.play();
      })
      .catch((error) => {
        console.error('Speech generation failed', error);
        const backendMessage = error.response?.data?.error;
        setMessage(backendMessage || 'Could not generate read aloud audio.');
      })
      .finally(() => {
        if (speechRunRef.current !== runId) return;
        if (audioRef.current?.paused) {
          setIsSpeaking(false);
        }
      });
  };

  const handleStopSpeaking = () => {
    speechRunRef.current += 1;
    audioRef.current?.pause();
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = '';
    }
    setIsSpeaking(false);
    setMessage('Speech stopped. Recognition is still ready for another run.');
  };

  const handleSpeechLanguageChange = (languageCode) => {
    setSpeechLang(languageCode);
  };

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <span className="eyebrow">Handwriting recognition</span>
          <h1>Turn handwritten images into usable text.</h1>
          <p>
            Upload a page or cropped word, recognize it, translate it, and listen
            to the result without leaving the screen.
          </p>
        </div>
        <div className="status-card">
          <span className="status-dot" />
          <strong>{loading ? 'Recognizing' : file ? 'Ready' : 'Waiting for image'}</strong>
          <small>
            {file
              ? 'You can run OCR again after stopping speech.'
              : 'Choose an image to begin.'}
          </small>
        </div>
      </section>

      <Upload file={file} previewUrl={previewUrl} onFileChange={handleFileChange} />

      <div className="action-row">
        <button type="button" onClick={handlePredict} disabled={!file || loading}>
          {loading ? 'Recognizing...' : `Recognize with ${ocrModels[ocrModel]}`}
        </button>
        <button
          type="button"
          onClick={() => runPrediction('custom')}
          className="ghost"
          disabled={!file || loading}
        >
          Custom HTR
        </button>
        <button
          type="button"
          onClick={() => runPrediction('hindi')}
          className="secondary"
          disabled={!file || loading}
        >
          Hindi HTR
        </button>
        <button
          type="button"
          onClick={() => runPrediction('trocr')}
          className="secondary"
          disabled={!file || loading}
        >
          TrOCR
        </button>
        <button type="button" onClick={handleCancel} className="ghost">
          Clear
        </button>
      </div>

      {(message || loading) && (
        <p className={`message ${loading ? 'pulse' : ''}`}>
          {message || 'Processing handwriting...'}
        </p>
      )}

      <Result
        text={text}
        translatedText={translatedText}
        targetLabel={targetLabel}
        modelLabel={lastModel || ocrModels[ocrModel]}
      />

      <Controls
        canUseText={Boolean(text)}
        isSpeaking={isSpeaking}
        languageOptions={languageOptions}
        speechLang={speechLang}
        speechSource={speechSource}
        targetLang={targetLang}
        hasTranslatedText={Boolean(translatedText)}
        onSpeechLangChange={handleSpeechLanguageChange}
        onSpeechSourceChange={setSpeechSource}
        onTargetLangChange={setTargetLang}
        onSpeak={handleSpeak}
        onStopSpeaking={handleStopSpeaking}
        onTranslate={handleTranslate}
        onAutocorrect={handleAutocorrect}
        translating={translating}
        autocorrecting={autocorrecting}
      />

      {history.length > 0 && (
        <section className="history">
          <div className="section-heading">
            <span className="eyebrow">Recent runs</span>
            <h2>Recognition history</h2>
          </div>
          <div className="history-list">
            {history.map((item) => (
              <button
                className="history-item"
                key={item.id}
                type="button"
                onClick={() => {
                  setText(item.text);
                  setTranslatedText('');
                  setSpeechSource('ocr');
                }}
              >
                <span>
                  <strong>{item.fileName}</strong>
                  <small>{item.time}</small>
                  <small>{item.model}</small>
                </span>
                <span>{item.text}</span>
              </button>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

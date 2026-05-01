
import axios from 'axios';

export const sendImage = async (file, model = 'custom', language = 'en-US') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', model);
  formData.append('language', language);

  const res = await axios.post('/predict', formData);
  return res.data;
};

export const translateText = async (text, targetLang) => {
  const trimmed = text.trim();

  if (!trimmed) {
    return '';
  }

  const url = 'https://api.mymemory.translated.net/get';
  const res = await axios.get(url, {
    params: {
      q: trimmed,
      langpair: `en|${targetLang}`,
    },
  });

  return res.data?.responseData?.translatedText || '';
};

export const synthesizeSpeech = async (text, lang) => {
  const res = await axios.post(
    '/tts',
    { text, lang },
    { responseType: 'blob' }
  );

  return URL.createObjectURL(res.data);
};

export const autocorrectText = async (text, language = 'en-US') => {
  const res = await axios.post('/autocorrect', { text, language });
  return res.data;
};

export const getTrocrStatus = async () => {
  const res = await axios.get('/trocr-status');
  return res.data;
};

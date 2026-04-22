export async function recognizeImage(
  imageBase64: string,
  mode: 'canvas' | 'photo' = 'canvas',
  mimeType: string = 'image/png'
) {
  // simulate processing delay
  await new Promise(res => setTimeout(res, 700));

  const samples = [
    "Hello World",
    "InkVoice is working",
    "Handwriting detected successfully",
    "This is a demo output",
    "AI recognition placeholder"
  ];

  const text = samples[Math.floor(Math.random() * samples.length)];

  return {
    id: Date.now().toString(),
    text,
    latency_ms: 700,
    model: "mock-engine"
  };
}
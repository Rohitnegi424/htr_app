import React, { useCallback, useState } from 'react';
import { View, Text, StyleSheet, Pressable, TextInput, ScrollView, ActivityIndicator, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

import { useTheme } from '../src/theme/ThemeContext';
import { getRecognitionEngineStatus, recognizeImage } from '../src/services/recognition';
import { speak, stopSpeaking } from '../src/services/tts';
import { addHistory } from '../src/services/history';

export default function ImageScreen() {
  const { colors, settings } = useTheme();
  const recognitionEngine = getRecognitionEngineStatus();
  const router = useRouter();
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [mime, setMime] = useState<string>('image/jpeg');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pickFromLibrary = async () => {
    try {
      Haptics.selectionAsync();
    } catch {}
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.9,
      base64: true,
    });
    if (!res.canceled && res.assets?.[0]) {
      const asset = res.assets[0];
      setImageUri(asset.uri);
      setImageBase64(asset.base64 ?? null);
      setMime(asset.mimeType || 'image/jpeg');
      setText('');
      setError(null);
    }
  };

  const captureWithCamera = async () => {
    try {
      Haptics.selectionAsync();
    } catch {}
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      setError('Camera permission denied.');
      return;
    }
    const res = await ImagePicker.launchCameraAsync({
      quality: 0.9,
      base64: true,
    });
    if (!res.canceled && res.assets?.[0]) {
      const asset = res.assets[0];
      setImageUri(asset.uri);
      setImageBase64(asset.base64 ?? null);
      setMime(asset.mimeType || 'image/jpeg');
      setText('');
      setError(null);
    }
  };

  const runRecognize = useCallback(async () => {
    if (!imageBase64) return;
    setLoading(true);
    setError(null);
    try {
      const result = await recognizeImage(imageBase64, 'photo', mime);
      const recognized = (result.text || '').trim();
      setText(recognized);
      if (recognized) {
        addHistory({ text: recognized, source: 'photo', latencyMs: result.latency_ms }).catch(() => {});
        if (settings.autoSpeak) speak(recognized, { rate: settings.ttsRate, pitch: settings.ttsPitch });
        try {
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        } catch {}
      } else {
        setError('No text detected in the image.');
      }
    } catch (e: any) {
      setError(e?.message ?? 'Recognition failed');
    } finally {
      setLoading(false);
    }
  }, [imageBase64, mime, settings.autoSpeak, settings.ttsPitch, settings.ttsRate]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.background }}>
      <View style={styles.topRow}>
        <Pressable
          onPress={() => {
            stopSpeaking();
            router.back();
          }}
          accessible
          accessibilityRole="button"
          accessibilityLabel="Go back"
          testID="img-back-btn"
          style={[styles.iconBtn, { backgroundColor: colors.surface, borderColor: colors.border }]}
        >
          <Ionicons name="chevron-back" size={22} color={colors.textPrimary} />
        </Pressable>
        <Text style={[styles.title, { color: colors.textPrimary }]}>Image recognition</Text>
        <View style={{ width: 44 }} />
      </View>

      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <View style={[styles.preview, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.previewImg} resizeMode="contain" />
          ) : (
            <View style={styles.placeholder}>
              <Ionicons name="scan-outline" size={44} color={colors.textSecondary} />
              <Text style={[styles.placeholderText, { color: colors.textSecondary }]}>
                Upload or capture a handwritten note to extract text.
              </Text>
            </View>
          )}
        </View>

        <View style={styles.btnRow}>
          <Pressable
            onPress={pickFromLibrary}
            accessible
            accessibilityRole="button"
            accessibilityLabel="Upload image from gallery"
            testID="pick-image-btn"
            style={({ pressed }) => [
              styles.primaryBtn,
              { backgroundColor: colors.surface, borderColor: colors.border, opacity: pressed ? 0.9 : 1 },
            ]}
          >
            <Ionicons name="images-outline" size={22} color={colors.textPrimary} />
            <Text style={[styles.primaryBtnLabel, { color: colors.textPrimary }]}>Gallery</Text>
          </Pressable>

          <Pressable
            onPress={captureWithCamera}
            accessible
            accessibilityRole="button"
            accessibilityLabel="Capture photo with camera"
            testID="capture-image-btn"
            style={({ pressed }) => [
              styles.primaryBtn,
              { backgroundColor: colors.surface, borderColor: colors.border, opacity: pressed ? 0.9 : 1 },
            ]}
          >
            <Ionicons name="camera-outline" size={22} color={colors.textPrimary} />
            <Text style={[styles.primaryBtnLabel, { color: colors.textPrimary }]}>Camera</Text>
          </Pressable>
        </View>

        <Pressable
          onPress={runRecognize}
          disabled={!imageBase64 || loading}
          accessible
          accessibilityRole="button"
          accessibilityLabel="Recognize text in image"
          testID="recognize-image-btn"
          style={({ pressed }) => [
            styles.ctaBtn,
            {
              backgroundColor: colors.primary,
              opacity: !imageBase64 || loading ? 0.5 : pressed ? 0.92 : 1,
            },
          ]}
        >
          {loading ? (
            <ActivityIndicator color="#0A0A0C" />
          ) : (
            <>
              <Ionicons name="sparkles" size={22} color="#0A0A0C" />
              <Text style={styles.ctaLabel}>Run recognition</Text>
            </>
          )}
        </Pressable>

        <Text style={[styles.engineStatus, { color: colors.textSecondary }]}>
          Engine: {recognitionEngine.label} {recognitionEngine.ready ? 'ready' : 'setup required'}
        </Text>
        <Text style={[styles.engineHelp, { color: colors.textSecondary }]}>
          Best results come from a cropped single word or short line on a plain background.
        </Text>

        {error ? (
          <Text style={[styles.err, { color: colors.primary }]} testID="img-error">
            {error}
          </Text>
        ) : null}

        <Text style={[styles.label, { color: colors.textSecondary }]}>Extracted text</Text>
        <TextInput
          value={text}
          onChangeText={setText}
          multiline
          placeholder="Recognized text appears here. You can edit it."
          placeholderTextColor={colors.textSecondary}
          accessibilityLabel="Editable recognized text"
          testID="extracted-text-input"
          style={[
            styles.input,
            {
              backgroundColor: colors.surface,
              borderColor: colors.border,
              color: colors.textPrimary,
            },
          ]}
        />

        <Pressable
          onPress={() => text && speak(text, { rate: settings.ttsRate, pitch: settings.ttsPitch })}
          disabled={!text}
          accessible
          accessibilityRole="button"
          accessibilityLabel="Read text aloud"
          testID="read-aloud-btn"
          style={({ pressed }) => [
            styles.ctaOutline,
            {
              borderColor: colors.primary,
              opacity: !text ? 0.4 : pressed ? 0.85 : 1,
            },
          ]}
        >
          <Ionicons name="volume-high" size={22} color={colors.primary} />
          <Text style={[styles.ctaOutlineLabel, { color: colors.primary }]}>Read aloud</Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  topRow: {
    paddingHorizontal: 18,
    paddingTop: 8,
    paddingBottom: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  iconBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: { fontSize: 18, fontWeight: '800' },
  content: { padding: 18, paddingBottom: 80, gap: 14 },
  preview: {
    minHeight: 220,
    borderRadius: 24,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  previewImg: { width: '100%', height: 260 },
  placeholder: { alignItems: 'center', padding: 24 },
  placeholderText: { marginTop: 10, textAlign: 'center', fontSize: 14, lineHeight: 20 },
  btnRow: { flexDirection: 'row', gap: 12 },
  primaryBtn: {
    flex: 1,
    minHeight: 60,
    borderRadius: 18,
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  primaryBtnLabel: { fontSize: 15, fontWeight: '700', marginLeft: 8 },
  ctaBtn: {
    minHeight: 64,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 10,
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 10 },
    elevation: 10,
  },
  ctaLabel: { color: '#0A0A0C', fontSize: 16, fontWeight: '800', marginLeft: 8 },
  engineStatus: { fontSize: 12, fontWeight: '600' },
  engineHelp: { fontSize: 12, lineHeight: 18, marginTop: -4 },
  err: { fontSize: 13, marginTop: 4 },
  label: { fontSize: 12, letterSpacing: 2, textTransform: 'uppercase', fontWeight: '700', marginTop: 4 },
  input: {
    minHeight: 140,
    borderRadius: 20,
    borderWidth: 1,
    padding: 16,
    fontSize: 17,
    textAlignVertical: 'top',
  },
  ctaOutline: {
    minHeight: 60,
    borderRadius: 20,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 10,
  },
  ctaOutlineLabel: { fontSize: 16, fontWeight: '800', marginLeft: 8 },
});

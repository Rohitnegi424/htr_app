import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, Pressable, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

import Canvas, { CanvasHandle } from '../src/components/Canvas';
import GlassPill from '../src/components/GlassPill';
import FAB from '../src/components/Fab';
import SheetMenu from '../src/components/SheetMenu';
import { useTheme } from '../src/theme/ThemeContext';
import { recognizeImage } from '../src/services/recognition';
import { speak, stopSpeaking } from '../src/services/tts';
import { addHistory } from '../src/services/history';

const IDLE_DEBOUNCE_MS = 900;

export default function WriteScreen() {
  const { colors, settings } = useTheme();
  const router = useRouter();
  const canvasRef = useRef<CanvasHandle>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightRef = useRef(false);
  const [recognized, setRecognized] = useState('');
  const [loading, setLoading] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const clearDebounce = () => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
  };

  useEffect(() => () => clearDebounce(), []);

  const runRecognition = useCallback(async () => {
    if (!canvasRef.current || canvasRef.current.isEmpty()) return;
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setLoading(true);
    setError(null);
    try {
      const base64 = await canvasRef.current.capturePng();
      const result = await recognizeImage(base64, 'canvas', 'image/png');
      const text = (result.text || '').trim();
      setRecognized(text);
      if (text) {
        if (settings.autoSpeak) speak(text, { rate: settings.ttsRate, pitch: settings.ttsPitch });
        addHistory({ text, source: 'canvas', latencyMs: result.latency_ms }).catch(() => {});
        try { Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success); } catch {}
      }
    } catch (e: any) {
      setError(e?.message ?? 'Recognition failed');
      setRecognized('');
    } finally {
      inFlightRef.current = false;
      setLoading(false);
    }
  }, [settings.autoSpeak, settings.ttsRate, settings.ttsPitch]);

  const onStrokeStart = useCallback(() => {
    clearDebounce();
    stopSpeaking();
  }, []);

  const onStrokeEnd = useCallback(() => {
    clearDebounce();
    debounceRef.current = setTimeout(() => {
      runRecognition();
    }, IDLE_DEBOUNCE_MS);
  }, [runRecognition]);

  const onClear = useCallback(() => {
    clearDebounce();
    stopSpeaking();
    canvasRef.current?.clear();
    setRecognized('');
    setError(null);
    try { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); } catch {}
  }, []);

  const onReplay = useCallback(() => {
    if (recognized) speak(recognized, { rate: settings.ttsRate, pitch: settings.ttsPitch });
  }, [recognized, settings.ttsRate, settings.ttsPitch]);

  return (
    <SafeAreaView edges={['top', 'left', 'right']} style={{ flex: 1, backgroundColor: colors.background }}>
      <View style={styles.header}>
        <View style={styles.brandRow}>
          <View style={[styles.brandDot, { backgroundColor: colors.primary }]} />
          <Text style={[styles.brand, { color: colors.textPrimary }]}>InkVoice</Text>
        </View>
        <Text style={[styles.subtle, { color: colors.textSecondary }]}>Write · Hear · Remember</Text>
      </View>

      <Animated.View entering={FadeInDown.duration(350)} style={styles.pillWrap}>
        <GlassPill text={recognized} loading={loading} />
        {error ? (
          <Text style={[styles.err, { color: colors.primary }]} testID="recognition-error">
            {error}
          </Text>
        ) : null}
      </Animated.View>

      <View style={styles.canvasWrap} testID="canvas-wrap">
        <Canvas
          ref={canvasRef}
          strokeColor={colors.canvasStroke}
          backgroundColor={colors.canvasBg}
          strokeWidth={settings.highContrast ? 8 : 6}
          onStrokeStart={onStrokeStart}
          onStrokeEnd={onStrokeEnd}
        />

        <Animated.View entering={FadeIn.delay(300)} style={styles.hintWrap} pointerEvents="none">
          {!recognized && !loading ? (
            <Text style={[styles.hint, { color: colors.textSecondary }]}>
              ✎ Write anywhere — pause to hear it aloud
            </Text>
          ) : null}
        </Animated.View>
      </View>

      <View style={styles.actions} pointerEvents="box-none">
        <Pressable
          onPress={onClear}
          accessible
          accessibilityRole="button"
          accessibilityLabel="Clear canvas"
          testID="clear-canvas-btn"
          style={({ pressed }) => [
            styles.secondaryBtn,
            {
              backgroundColor: colors.surface,
              borderColor: colors.border,
              opacity: pressed ? 0.85 : 1,
            },
          ]}
        >
          <Ionicons name="trash-outline" size={22} color={colors.textPrimary} />
          <Text style={[styles.secondaryLabel, { color: colors.textPrimary }]}>Clear</Text>
        </Pressable>

        <Pressable
          onPress={onReplay}
          disabled={!recognized}
          accessible
          accessibilityRole="button"
          accessibilityLabel="Read recognized text aloud again"
          testID="replay-tts-btn"
          style={({ pressed }) => [
            styles.secondaryBtn,
            {
              backgroundColor: colors.surface,
              borderColor: colors.border,
              opacity: !recognized ? 0.4 : pressed ? 0.85 : 1,
            },
          ]}
        >
          <Ionicons name="volume-high-outline" size={22} color={colors.textPrimary} />
          <Text style={[styles.secondaryLabel, { color: colors.textPrimary }]}>Replay</Text>
        </Pressable>
      </View>

      <FAB onPress={() => setSheetOpen(true)} icon="apps" label="Open quick actions menu" testID="main-fab" />

      <SheetMenu
        visible={sheetOpen}
        onClose={() => setSheetOpen(false)}
        items={[
          {
            key: 'image',
            label: 'Upload / capture image',
            hint: 'Recognize handwriting from a photo',
            icon: 'image-outline',
            onPress: () => router.push('/image'),
            testID: 'sheet-image',
          },
          {
            key: 'history',
            label: 'History',
            hint: 'Your saved recognitions',
            icon: 'time-outline',
            onPress: () => router.push('/history'),
            testID: 'sheet-history',
          },
          {
            key: 'settings',
            label: 'Settings',
            hint: 'Theme, contrast, voice',
            icon: 'settings-outline',
            onPress: () => router.push('/settings'),
            testID: 'sheet-settings',
          },
        ]}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  header: { paddingHorizontal: 22, paddingTop: 10, paddingBottom: 6 },
  brandRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  brandDot: { width: 10, height: 10, borderRadius: 5, marginRight: 10 },
  brand: { fontSize: 22, fontWeight: '800', letterSpacing: 0.2 },
  subtle: { fontSize: 13, marginTop: 4, letterSpacing: 2, textTransform: 'uppercase', fontWeight: '600' },
  pillWrap: { paddingHorizontal: 18, paddingTop: 12, paddingBottom: 6 },
  err: { marginTop: 8, marginLeft: 6, fontSize: 13 },
  canvasWrap: { flex: 1, marginHorizontal: 16, marginTop: 10, marginBottom: 110, borderRadius: 28, overflow: 'hidden' },
  hintWrap: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
  hint: { fontSize: 16, fontWeight: '500' },
  actions: {
    position: 'absolute',
    bottom: 36,
    left: 20,
    flexDirection: 'row',
    gap: 12,
  },
  secondaryBtn: {
    minHeight: 60,
    minWidth: 60,
    paddingHorizontal: 15,
    borderRadius: 22,
    borderWidth: 1,
    // marginRight: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    ...Platform.select({
      ios: { shadowColor: '#000', shadowOpacity: 0.25, shadowRadius: 16, shadowOffset: { width: 0, height: 8 } },
      android: { elevation: 6 },
      default: {},
    }),
  },
  secondaryLabel: { fontSize: 15, fontWeight: '700', marginLeft: 8 },
});



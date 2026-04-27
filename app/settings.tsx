import React from 'react';
import { View, Text, StyleSheet, Pressable, Switch, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Slider from '@react-native-community/slider';

import { useTheme } from '../src/theme/ThemeContext';
import { getRecognitionEngineStatus, getRecognitionHealth, getRecognitionSetupSteps } from '../src/services/recognition';
import { speak, stopSpeaking } from '../src/services/tts';
import { clearHistory } from '../src/services/history';

export default function SettingsScreen() {
  const { colors, settings, setMode, toggleHighContrast, update } = useTheme();
  const recognitionEngine = getRecognitionEngineStatus();
  const setupSteps = getRecognitionSetupSteps();
  const [health, setHealth] = React.useState<{ ok: boolean; error?: string | null; charset_size?: number } | null>(null);
  const router = useRouter();

  React.useEffect(() => {
    getRecognitionHealth().then(setHealth).catch(() => {
      setHealth({ ok: false, error: 'Failed to reach the local HTR backend.' });
    });
  }, []);

  const Row = ({
    title,
    description,
    right,
    testID,
  }: {
    title: string;
    description?: string;
    right: React.ReactNode;
    testID?: string;
  }) => (
    <View
      style={[styles.row, { borderColor: colors.border }]}
      accessible
      accessibilityLabel={title}
      testID={testID}
    >
      <View style={{ flex: 1, paddingRight: 12 }}>
        <Text style={[styles.rowTitle, { color: colors.textPrimary }]}>{title}</Text>
        {description ? <Text style={[styles.rowDesc, { color: colors.textSecondary }]}>{description}</Text> : null}
      </View>
      {right}
    </View>
  );

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
          testID="settings-back-btn"
          style={[styles.iconBtn, { backgroundColor: colors.surface, borderColor: colors.border }]}
        >
          <Ionicons name="chevron-back" size={22} color={colors.textPrimary} />
        </Pressable>
        <Text style={[styles.title, { color: colors.textPrimary }]}>Settings</Text>
        <View style={{ width: 44 }} />
      </View>

      <ScrollView contentContainerStyle={{ padding: 18, paddingBottom: 60, gap: 8 }}>
        <Text style={[styles.section, { color: colors.textSecondary }]}>Appearance</Text>

        <Row
          title="Theme"
          description="Dark or light canvas"
          testID="settings-theme-row"
          right={
            <View style={[styles.segWrap, { borderColor: colors.border, backgroundColor: colors.surface }]}>
              {(['dark', 'light'] as const).map((mode) => {
                const active = settings.mode === mode && !settings.highContrast;
                return (
                  <Pressable
                    key={mode}
                    onPress={() => setMode(mode)}
                    accessibilityRole="button"
                    accessibilityLabel={`Set theme to ${mode}`}
                    testID={`settings-theme-${mode}`}
                    style={[styles.segItem, active && { backgroundColor: colors.primary }]}
                  >
                    <Text style={{ color: active ? '#0A0A0C' : colors.textPrimary, fontWeight: '700' }}>
                      {mode[0].toUpperCase() + mode.slice(1)}
                    </Text>
                  </Pressable>
                );
              })}
            </View>
          }
        />

        <Row
          title="High contrast"
          description="Pure black background, yellow accents (WCAG AAA)"
          testID="settings-highcontrast-row"
          right={
            <Switch
              value={settings.highContrast}
              onValueChange={toggleHighContrast}
              trackColor={{ true: colors.primary, false: colors.border }}
              thumbColor="#FFFFFF"
              accessibilityLabel="High contrast mode"
              testID="settings-highcontrast-switch"
            />
          }
        />

        <Text style={[styles.section, { color: colors.textSecondary }]}>Voice</Text>

        <Row
          title="Auto speak"
          description="Read text aloud after recognition"
          testID="settings-autospeak-row"
          right={
            <Switch
              value={settings.autoSpeak}
              onValueChange={(value) => update({ autoSpeak: value })}
              trackColor={{ true: colors.primary, false: colors.border }}
              thumbColor="#FFFFFF"
              testID="settings-autospeak-switch"
            />
          }
        />

        <View style={[styles.sliderCard, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.rowTitle, { color: colors.textPrimary }]}>Speech rate</Text>
          <Text style={[styles.rowDesc, { color: colors.textSecondary }]}>{settings.ttsRate.toFixed(2)}x</Text>
          <Slider
            minimumValue={0.5}
            maximumValue={1.6}
            step={0.05}
            value={settings.ttsRate}
            onValueChange={(value) => update({ ttsRate: Number(value.toFixed(2)) })}
            minimumTrackTintColor={colors.primary}
            maximumTrackTintColor={colors.border}
            thumbTintColor={colors.primary}
            accessibilityLabel="Adjust speech rate"
            testID="settings-rate-slider"
          />
        </View>

        <View style={[styles.sliderCard, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.rowTitle, { color: colors.textPrimary }]}>Speech pitch</Text>
          <Text style={[styles.rowDesc, { color: colors.textSecondary }]}>{settings.ttsPitch.toFixed(2)}</Text>
          <Slider
            minimumValue={0.5}
            maximumValue={1.6}
            step={0.05}
            value={settings.ttsPitch}
            onValueChange={(value) => update({ ttsPitch: Number(value.toFixed(2)) })}
            minimumTrackTintColor={colors.primary}
            maximumTrackTintColor={colors.border}
            thumbTintColor={colors.primary}
            accessibilityLabel="Adjust speech pitch"
            testID="settings-pitch-slider"
          />
        </View>

        <Pressable
          onPress={() =>
            speak('InkVoice is ready. Write naturally on the canvas and I will read your words aloud.', {
              rate: settings.ttsRate,
              pitch: settings.ttsPitch,
            })
          }
          accessible
          accessibilityRole="button"
          accessibilityLabel="Test voice"
          testID="settings-test-voice-btn"
          style={({ pressed }) => [
            styles.testBtn,
            { borderColor: colors.primary, opacity: pressed ? 0.9 : 1 },
          ]}
        >
          <Ionicons name="play" size={20} color={colors.primary} />
          <Text style={[styles.testLabel, { color: colors.primary }]}>Test voice</Text>
        </Pressable>

        <Text style={[styles.section, { color: colors.textSecondary }]}>Data</Text>

        <Pressable
          onPress={async () => {
            await clearHistory();
          }}
          accessible
          accessibilityRole="button"
          accessibilityLabel="Clear saved history"
          testID="settings-clear-history-btn"
          style={({ pressed }) => [
            styles.dangerBtn,
            { borderColor: colors.border, backgroundColor: colors.surface, opacity: pressed ? 0.9 : 1 },
          ]}
        >
          <Ionicons name="trash-outline" size={20} color={colors.textPrimary} />
          <Text style={[styles.testLabel, { color: colors.textPrimary }]}>Clear history</Text>
        </Pressable>

        <Text style={[styles.section, { color: colors.textSecondary }]}>Recognition engine</Text>
        <View style={[styles.sliderCard, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.rowTitle, { color: colors.textPrimary }]}>{recognitionEngine.label}</Text>
          <Text style={[styles.rowDesc, { color: colors.textSecondary }]}>
            Status: {health ? (health.ok ? 'backend reachable' : 'backend not ready') : 'checking'}.
          </Text>
          <Text style={[styles.rowDesc, { color: colors.textSecondary }]}>{recognitionEngine.detail}</Text>
          {health?.charset_size ? (
            <Text style={[styles.rowDesc, { color: colors.textSecondary }]}>Charset size: {health.charset_size}</Text>
          ) : null}
          {health?.error ? <Text style={[styles.rowDesc, { color: colors.primary }]}>{health.error}</Text> : null}
          {setupSteps.map((step, index) => (
            <Text key={step} style={[styles.stepText, { color: colors.textSecondary }]}>
              {index + 1}. {step}
            </Text>
          ))}
        </View>
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
  iconBtn: { width: 44, height: 44, borderRadius: 22, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  title: { fontSize: 18, fontWeight: '800' },
  section: {
    fontSize: 11,
    letterSpacing: 2,
    textTransform: 'uppercase',
    fontWeight: '800',
    marginTop: 18,
    marginBottom: 6,
    marginLeft: 4,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 16,
    paddingHorizontal: 4,
    borderBottomWidth: StyleSheet.hairlineWidth,
    minHeight: 64,
  },
  rowTitle: { fontSize: 16, fontWeight: '700' },
  rowDesc: { fontSize: 13, marginTop: 4, lineHeight: 18 },
  stepText: { fontSize: 13, marginTop: 8, lineHeight: 18 },
  segWrap: { flexDirection: 'row', borderRadius: 14, borderWidth: 1, overflow: 'hidden' },
  segItem: { minWidth: 68, minHeight: 40, paddingHorizontal: 14, alignItems: 'center', justifyContent: 'center' },
  sliderCard: { padding: 16, borderRadius: 20, borderWidth: 1, marginTop: 8 },
  testBtn: {
    minHeight: 56,
    borderRadius: 18,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 10,
    marginTop: 12,
  },
  testLabel: { fontSize: 15, fontWeight: '800', marginLeft: 8 },
  dangerBtn: {
    minHeight: 56,
    borderRadius: 18,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 10,
    marginTop: 4,
  },
});

import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { BlurView } from 'expo-blur';
import { useTheme } from '../theme/ThemeContext';

type Props = {
  text: string;
  loading?: boolean;
  placeholder?: string;
};

export default function GlassPill({ text, loading, placeholder = 'Write below — I will read it aloud' }: Props) {
  const { colors, settings } = useTheme();
  const showPlaceholder = !text && !loading;
  const useBlur = !settings.highContrast;

  const Content = (
    <View
      style={[
        styles.inner,
        {
          backgroundColor: useBlur ? 'transparent' : colors.surface,
          borderColor: colors.glassBorder,
        },
      ]}
      accessible
      accessibilityLiveRegion="polite"
      accessibilityRole="text"
      accessibilityLabel={loading ? 'Recognizing handwriting' : text || placeholder}
      testID="recognized-text-display"
    >
      {loading ? (
        <View style={styles.row}>
          <ActivityIndicator size="small" color={colors.primary} />
          <Text style={[styles.label, { color: colors.textSecondary }]}>Recognizing…</Text>
        </View>
      ) : (
        <Text
          style={[
            styles.text,
            { color: showPlaceholder ? colors.textSecondary : colors.textPrimary },
          ]}
          numberOfLines={3}
        >
          {showPlaceholder ? placeholder : text}
        </Text>
      )}
    </View>
  );

  if (useBlur) {
    return (
      <BlurView
        intensity={60}
        tint={colors.blurTint}
        style={[styles.wrap, { borderColor: colors.glassBorder, backgroundColor: colors.glassBg }]}
      >
        {Content}
      </BlurView>
    );
  }

  return (
    <View style={[styles.wrap, { borderColor: colors.border, backgroundColor: colors.background }]}>
      {Content}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    borderRadius: 28,
    borderWidth: 1,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOpacity: 0.35,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 10 },
    elevation: 14,
  },
  inner: {
    paddingVertical: 18,
    paddingHorizontal: 22,
    borderRadius: 28,
    borderWidth: 0,
  },
  text: {
    fontSize: 22,
    lineHeight: 28,
    fontWeight: '600',
    textAlign: 'center',
  },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 12 },
  label: { marginLeft: 10, fontSize: 16 },
});



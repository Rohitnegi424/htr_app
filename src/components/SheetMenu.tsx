import React, { useEffect } from 'react';
import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { useTheme } from '../theme/ThemeContext';

export type SheetItem = {
  key: string;
  label: string;
  hint?: string;
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
  testID?: string;
};

type Props = {
  visible: boolean;
  onClose: () => void;
  items: SheetItem[];
};

export default function SheetMenu({ visible, onClose, items }: Props) {
  const { colors, settings } = useTheme();
  const translateY = useSharedValue(400);
  const opacity = useSharedValue(0);

  useEffect(() => {
    if (visible) {
      opacity.value = withTiming(1, { duration: 180 });
      translateY.value = withSpring(0, { stiffness: 220, damping: 22 });
    } else {
      opacity.value = withTiming(0, { duration: 140 });
      translateY.value = withTiming(400, { duration: 180 });
    }
  }, [visible, opacity, translateY]);

  const sheetStyle = useAnimatedStyle(() => ({ transform: [{ translateY: translateY.value }] }));
  const scrimStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));

  const dismiss = () => {
    try { Haptics.selectionAsync(); } catch {}
    onClose();
  };

  const handleItem = (fn: () => void) => {
    try { Haptics.selectionAsync(); } catch {}
    onClose();
    setTimeout(fn, 180);
  };

  return (
    <Modal visible={visible} transparent animationType="none" onRequestClose={onClose}>
      <Animated.View style={[StyleSheet.absoluteFill, scrimStyle, { backgroundColor: colors.scrim }]}>
        <Pressable style={StyleSheet.absoluteFill} onPress={dismiss} accessibilityLabel="Close menu" testID="sheet-scrim" />
      </Animated.View>
      <View style={styles.anchor} pointerEvents="box-none">
        <Animated.View
          style={[
            styles.sheet,
            sheetStyle,
            {
              backgroundColor: settings.highContrast ? colors.background : colors.surfaceElevated,
              borderColor: colors.glassBorder,
            },
          ]}
          testID="navigation-bottom-sheet"
        >
          {!settings.highContrast && (
            <BlurView intensity={30} tint={colors.blurTint} style={StyleSheet.absoluteFill} />
          )}
          <View style={[styles.handle, { backgroundColor: colors.textSecondary }]} />
          <Text style={[styles.title, { color: colors.textPrimary }]}>Quick actions</Text>
          {items.map((it) => (
            <Pressable
              key={it.key}
              onPress={() => handleItem(it.onPress)}
              accessible
              accessibilityRole="button"
              accessibilityLabel={it.label}
              accessibilityHint={it.hint}
              testID={it.testID ?? `sheet-item-${it.key}`}
              style={({ pressed }) => [
                styles.row,
                {
                  backgroundColor: pressed ? colors.surface : 'transparent',
                  borderColor: colors.border,
                },
              ]}
            >
              <View style={[styles.iconWrap, { backgroundColor: colors.surface, borderColor: colors.border }]}>
                <Ionicons name={it.icon} size={26} color={colors.primary} />
              </View>
              <View style={styles.textCol}>
                <Text style={[styles.rowLabel, { color: colors.textPrimary }]}>{it.label}</Text>
                {it.hint ? <Text style={[styles.rowHint, { color: colors.textSecondary }]}>{it.hint}</Text> : null}
              </View>
              <Ionicons name="chevron-forward" size={22} color={colors.textSecondary} />
            </Pressable>
          ))}
        </Animated.View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  anchor: { flex: 1, justifyContent: 'flex-end' },
  sheet: {
    padding: 20,
    paddingBottom: 36,
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    borderWidth: 1,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOpacity: 0.45,
    shadowRadius: 30,
    shadowOffset: { width: 0, height: -10 },
    elevation: 20,
  },
  handle: { alignSelf: 'center', width: 56, height: 6, borderRadius: 3, opacity: 0.4, marginBottom: 18 },
  title: { fontSize: 20, fontWeight: '700', marginBottom: 14, marginLeft: 6 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 20,
    borderWidth: 1,
    marginBottom: 10,
    minHeight: 72,
  },
  iconWrap: {
    width: 52,
    height: 52,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    marginRight: 16,
  },
  textCol: { flex: 1 },
  rowLabel: { fontSize: 18, fontWeight: '700' },
  rowHint: { fontSize: 14, marginTop: 2 },
});


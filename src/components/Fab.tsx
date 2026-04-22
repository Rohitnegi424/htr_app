import React from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';
import { useTheme } from '../theme/ThemeContext';

type Props = {
  onPress: () => void;
  icon?: keyof typeof Ionicons.glyphMap;
  label?: string;
  testID?: string;
};

export default function FAB({ onPress, icon = 'menu', label = 'Open menu', testID = 'main-fab' }: Props) {
  const { colors } = useTheme();
  const scale = useSharedValue(1);

  const style = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));

  const handlePress = () => {
    try { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium); } catch {}
    onPress();
  };

  return (
    <Animated.View style={[styles.wrap, style]}>
      <Pressable
        onPressIn={() => { scale.value = withSpring(0.92, { stiffness: 260, damping: 18 }); }}
        onPressOut={() => { scale.value = withSpring(1, { stiffness: 260, damping: 18 }); }}
        onPress={handlePress}
        accessible
        accessibilityRole="button"
        accessibilityLabel={label}
        testID={testID}
        style={({ pressed }) => [
          styles.btn,
          {
            backgroundColor: colors.primary,
            shadowColor: colors.primary,
            opacity: pressed ? 0.92 : 1,
          },
        ]}
      >
        <View style={styles.inner}>
          <Ionicons name={icon} size={30} color="#0A0A0C" />
        </View>
      </Pressable>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  wrap: { position: 'absolute', bottom: 28, right: 24 },
  btn: {
    width: 72,
    height: 72,
    borderRadius: 36,
    alignItems: 'center',
    justifyContent: 'center',
    shadowOpacity: 0.4,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 12 },
    elevation: 14,
  },
  inner: { alignItems: 'center', justifyContent: 'center' },
});



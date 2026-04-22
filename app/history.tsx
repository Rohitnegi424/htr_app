import React, { useCallback, useEffect, useState } from 'react';
import { View, Text, StyleSheet, Pressable, FlatList, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter, useFocusEffect } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { useTheme } from '../src/theme/ThemeContext';
import { getHistory, removeHistory, clearHistory, HistoryItem } from '../src/services/history';
import { speak, stopSpeaking } from '../src/services/tts';

function formatTime(ts: number) {
  const d = new Date(ts);
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export default function HistoryScreen() {
  const { colors, settings } = useTheme();
  const router = useRouter();
  const [items, setItems] = useState<HistoryItem[]>([]);

  const load = useCallback(async () => {
    const list = await getHistory();
    setItems(list);
  }, []);

  useFocusEffect(
    useCallback(() => {
      load();
      return () => stopSpeaking();
    }, [load])
  );

  const onDelete = useCallback(async (id: string) => {
    await removeHistory(id);
    load();
  }, [load]);

  const onClearAll = useCallback(() => {
    Alert.alert('Clear all history?', 'This cannot be undone.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Clear',
        style: 'destructive',
        onPress: async () => {
          await clearHistory();
          load();
        },
      },
    ]);
  }, [load]);

  const renderItem = ({ item }: { item: HistoryItem }) => (
    <Pressable
      onPress={() => speak(item.text, { rate: settings.ttsRate, pitch: settings.ttsPitch })}
      accessible accessibilityRole="button" accessibilityLabel={`Play: ${item.text}`}
      testID={`history-item-${item.id}`}
      style={({ pressed }) => [
        styles.card,
        {
          backgroundColor: colors.surface,
          borderColor: colors.border,
          opacity: pressed ? 0.85 : 1,
        },
      ]}
    >
      <View style={[styles.iconWrap, { backgroundColor: colors.surfaceElevated, borderColor: colors.border }]}>
        <Ionicons name={item.source === 'canvas' ? 'create-outline' : 'image-outline'} size={22} color={colors.primary} />
      </View>
      <View style={{ flex: 1 }}>
        <Text numberOfLines={2} style={[styles.cardText, { color: colors.textPrimary }]}>{item.text}</Text>
        <Text style={[styles.cardMeta, { color: colors.textSecondary }]}>
          {formatTime(item.createdAt)} · {item.source === 'canvas' ? 'Handwritten' : 'Photo'}
        </Text>
      </View>
      <Pressable
        onPress={() => onDelete(item.id)}
        accessible accessibilityRole="button" accessibilityLabel="Delete entry"
        testID={`history-delete-${item.id}`}
        hitSlop={12}
        style={styles.delBtn}
      >
        <Ionicons name="trash-outline" size={20} color={colors.textSecondary} />
      </Pressable>
    </Pressable>
  );

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: colors.background }}>
      <View style={styles.topRow}>
        <Pressable
          onPress={() => { stopSpeaking(); router.back(); }}
          accessible accessibilityRole="button" accessibilityLabel="Go back"
          testID="history-back-btn"
          style={[styles.iconBtn, { backgroundColor: colors.surface, borderColor: colors.border }]}
        >
          <Ionicons name="chevron-back" size={22} color={colors.textPrimary} />
        </Pressable>
        <Text style={[styles.title, { color: colors.textPrimary }]}>History</Text>
        <Pressable
          onPress={onClearAll}
          disabled={items.length === 0}
          accessible accessibilityRole="button" accessibilityLabel="Clear all history"
          testID="history-clear-all-btn"
          style={[
            styles.iconBtn,
            { backgroundColor: colors.surface, borderColor: colors.border, opacity: items.length === 0 ? 0.4 : 1 },
          ]}
        >
          <Ionicons name="trash-outline" size={20} color={colors.textPrimary} />
        </Pressable>
      </View>

      {items.length === 0 ? (
        <View style={styles.empty} testID="history-empty">
          <Ionicons name="book-outline" size={46} color={colors.textSecondary} />
          <Text style={[styles.emptyTitle, { color: colors.textPrimary }]}>No entries yet</Text>
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
            Your recognized notes will appear here. Start writing on the canvas.
          </Text>
        </View>
      ) : (
        <FlatList
          data={items}
          keyExtractor={(i) => i.id}
          renderItem={renderItem}
          contentContainerStyle={{ padding: 18, paddingBottom: 60, gap: 12 }}
          ItemSeparatorComponent={() => <View style={{ height: 10 }} />}
          testID="history-list"
        />
      )}
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
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 22,
    borderWidth: 1,
    gap: 14,
  },
  iconWrap: { width: 48, height: 48, borderRadius: 14, alignItems: 'center', justifyContent: 'center', borderWidth: 1 },
  cardText: { fontSize: 16, fontWeight: '600' },
  cardMeta: { fontSize: 12, marginTop: 4 },
  delBtn: { padding: 8, borderRadius: 14 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32, gap: 10 },
  emptyTitle: { fontSize: 20, fontWeight: '800', marginTop: 10 },
  emptyText: { fontSize: 14, textAlign: 'center', lineHeight: 20 },
});



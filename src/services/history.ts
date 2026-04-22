import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = '@inkvoice/history/v1';

export type HistoryItem = {
  id: string;
  text: string;
  source: 'canvas' | 'photo';
  createdAt: number;
  latencyMs?: number;
};

export async function getHistory(): Promise<HistoryItem[]> {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryItem[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export async function addHistory(item: Omit<HistoryItem, 'id' | 'createdAt'> & { id?: string; createdAt?: number }) {
  const list = await getHistory();
  const full: HistoryItem = {
    id: item.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: item.createdAt ?? Date.now(),
    text: item.text,
    source: item.source,
    latencyMs: item.latencyMs,
  };
  const next = [full, ...list].slice(0, 200);
  await AsyncStorage.setItem(KEY, JSON.stringify(next));
  return full;
}

export async function removeHistory(id: string) {
  const list = await getHistory();
  const next = list.filter((i) => i.id !== id);
  await AsyncStorage.setItem(KEY, JSON.stringify(next));
}

export async function clearHistory() {
  await AsyncStorage.removeItem(KEY);
}



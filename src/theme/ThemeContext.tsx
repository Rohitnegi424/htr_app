import React, { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export type ThemeMode = 'dark' | 'light';

export type Settings = {
  mode: ThemeMode;
  highContrast: boolean;
  ttsRate: number;
  ttsPitch: number;
  autoSpeak: boolean;
};

const defaultSettings: Settings = {
  mode: 'dark',
  highContrast: false,
  ttsRate: 1.0,
  ttsPitch: 1.0,
  autoSpeak: true,
};

const STORAGE_KEY = '@inkvoice/settings/v1';

type Palette = {
  background: string;
  surface: string;
  surfaceElevated: string;
  primary: string;
  primaryActive: string;
  textPrimary: string;
  textSecondary: string;
  border: string;
  glassBg: string;
  glassBorder: string;
  scrim: string;
  canvasStroke: string;
  canvasBg: string;
  blurTint: 'dark' | 'light';
};

const darkPalette: Palette = {
  background: '#050505',
  surface: '#121214',
  surfaceElevated: '#1A1A1E',
  primary: '#FFB800',
  primaryActive: '#E6A600',
  textPrimary: '#FFFFFF',
  textSecondary: '#A1A1AA',
  border: '#27272A',
  glassBg: 'rgba(18,18,20,0.7)',
  glassBorder: 'rgba(255,255,255,0.1)',
  scrim: 'rgba(0,0,0,0.5)',
  canvasStroke: '#FFB800',
  canvasBg: '#0A0A0C',
  blurTint: 'dark',
};

const lightPalette: Palette = {
  background: '#FAFAFA',
  surface: '#FFFFFF',
  surfaceElevated: '#F4F4F5',
  primary: '#D97706',
  primaryActive: '#B45309',
  textPrimary: '#09090B',
  textSecondary: '#52525B',
  border: '#E4E4E7',
  glassBg: 'rgba(255,255,255,0.8)',
  glassBorder: 'rgba(0,0,0,0.06)',
  scrim: 'rgba(0,0,0,0.3)',
  canvasStroke: '#111111',
  canvasBg: '#FFFFFF',
  blurTint: 'light',
};

const highContrastPalette: Palette = {
  background: '#000000',
  surface: '#000000',
  surfaceElevated: '#000000',
  primary: '#FFFF00',
  primaryActive: '#FFFF00',
  textPrimary: '#FFFFFF',
  textSecondary: '#FFFFFF',
  border: '#FFFFFF',
  glassBg: '#000000',
  glassBorder: '#FFFFFF',
  scrim: 'rgba(0,0,0,0.9)',
  canvasStroke: '#FFFF00',
  canvasBg: '#000000',
  blurTint: 'dark',
};

type Ctx = {
  settings: Settings;
  colors: Palette;
  setMode: (mode: ThemeMode) => void;
  toggleHighContrast: () => void;
  update: (patch: Partial<Settings>) => void;
};

const ThemeContext = createContext<Ctx | null>(null);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const raw = await AsyncStorage.getItem(STORAGE_KEY);
        if (raw) setSettings({ ...defaultSettings, ...JSON.parse(raw) });
      } catch {}
      setHydrated(true);
    })();
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(settings)).catch(() => {});
  }, [settings, hydrated]);

  const colors = useMemo(() => {
    if (settings.highContrast) return highContrastPalette;
    return settings.mode === 'dark' ? darkPalette : lightPalette;
  }, [settings.mode, settings.highContrast]);

  const setMode = useCallback((mode: ThemeMode) => setSettings((s) => ({ ...s, mode })), []);
  const toggleHighContrast = useCallback(
    () => setSettings((s) => ({ ...s, highContrast: !s.highContrast })),
    []
  );
  const update = useCallback((patch: Partial<Settings>) => setSettings((s) => ({ ...s, ...patch })), []);

  const value = useMemo(
    () => ({ settings, colors, setMode, toggleHighContrast, update }),
    [settings, colors, setMode, toggleHighContrast, update]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

export const useTheme = () => {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
};



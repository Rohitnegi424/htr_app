import React, { forwardRef, useImperativeHandle, useRef, useState, useCallback } from 'react';
import { View, PanResponder, StyleSheet, LayoutChangeEvent } from 'react-native';
import Svg, { Path, Rect } from 'react-native-svg';
import { captureRef } from 'react-native-view-shot';

export type CanvasHandle = {
  clear: () => void;
  isEmpty: () => boolean;
  capturePng: () => Promise<string>;
};

type Props = {
  strokeColor: string;
  backgroundColor: string;
  strokeWidth?: number;
  onStrokeStart?: () => void;
  onStrokeEnd?: () => void;
};

const Canvas = forwardRef<CanvasHandle, Props>(
  ({ strokeColor, backgroundColor, strokeWidth = 6, onStrokeStart, onStrokeEnd }, ref) => {
    const [paths, setPaths] = useState<string[]>([]);
    const [current, setCurrent] = useState<string>('');
    const pathsRef = useRef<string[]>([]);
    const currentRef = useRef<string>('');
    const shotRef = useRef<View>(null);
    const sizeRef = useRef({ w: 0, h: 0 });

    const onLayout = (e: LayoutChangeEvent) => {
      sizeRef.current = { w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height };
    };

    const responder = useRef(
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderGrant: (evt) => {
          onStrokeStart?.();
          const { locationX, locationY } = evt.nativeEvent;
          const d = `M${locationX.toFixed(1)},${locationY.toFixed(1)}`;
          currentRef.current = d;
          setCurrent(d);
        },
        onPanResponderMove: (evt) => {
          const { locationX, locationY } = evt.nativeEvent;
          const d = `${currentRef.current} L${locationX.toFixed(1)},${locationY.toFixed(1)}`;
          currentRef.current = d;
          setCurrent(d);
        },
        onPanResponderRelease: () => {
          if (currentRef.current) {
            pathsRef.current = [...pathsRef.current, currentRef.current];
            setPaths(pathsRef.current);
          }
          currentRef.current = '';
          setCurrent('');
          onStrokeEnd?.();
        },
        onPanResponderTerminate: () => {
          if (currentRef.current) {
            pathsRef.current = [...pathsRef.current, currentRef.current];
            setPaths(pathsRef.current);
          }
          currentRef.current = '';
          setCurrent('');
          onStrokeEnd?.();
        },
      })
    ).current;

    const clear = useCallback(() => {
      pathsRef.current = [];
      currentRef.current = '';
      setPaths([]);
      setCurrent('');
    }, []);

    const isEmpty = useCallback(() => pathsRef.current.length === 0 && !currentRef.current, []);

    const capturePng = useCallback(async () => {
      const uri = await captureRef(shotRef, {
        format: 'png',
        quality: 0.9,
        result: 'base64',
      });
      return uri as string;
    }, []);

    useImperativeHandle(ref, () => ({ clear, isEmpty, capturePng }), [clear, isEmpty, capturePng]);

    return (
      <View
        ref={shotRef}
        collapsable={false}
        onLayout={onLayout}
        style={[styles.container, { backgroundColor }]}
        {...responder.panHandlers}
        accessible
        accessibilityRole="image"
        accessibilityLabel="Drawing canvas. Write anywhere with your finger."
        accessibilityHint="Strokes are recognized as text after you pause."
        testID="drawing-canvas"
      >
        <Svg style={StyleSheet.absoluteFill} width="100%" height="100%">
          <Rect x={0} y={0} width="100%" height="100%" fill={backgroundColor} />
          {paths.map((d, idx) => (
            <Path
              key={idx}
              d={d}
              stroke={strokeColor}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          ))}
          {current ? (
            <Path
              d={current}
              stroke={strokeColor}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          ) : null}
        </Svg>
      </View>
    );
  }
);

Canvas.displayName = 'Canvas';

const styles = StyleSheet.create({
  container: { flex: 1, overflow: 'hidden' },
});

export default Canvas;



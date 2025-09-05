import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'app_text_theme.dart';
import 'palette.dart';

class AppTheme {
  AppTheme._();

  // Color schemes using Material Design 3
  static final ColorScheme _lightColorScheme =
      ColorScheme.fromSeed(seedColor: Palette.primaryBlue, brightness: Brightness.light).copyWith(
        surface: Palette.lightBackground,
        onSurface: Palette.lightText,
        surfaceContainer: Palette.lightSurface,
        outline: Palette.lightBorder,
        outlineVariant: Palette.lightBorderSecondary,
        onSurfaceVariant: Palette.lightTextMuted,
        surfaceContainerLow: Palette.lightSurfaceSecondary,
        surfaceContainerHigh: Palette.lightDivider,
      );

  static final ColorScheme _darkColorScheme =
      ColorScheme.fromSeed(seedColor: Palette.primaryBlue, brightness: Brightness.dark).copyWith(
        surface: Palette.darkBackground,
        onSurface: Palette.darkText,
        surfaceContainer: Palette.darkSurface,
        outline: Palette.darkBorder,
        outlineVariant: Palette.darkBorderSecondary,
        onSurfaceVariant: Palette.darkTextMuted,
        surfaceContainerLow: Palette.darkSurfaceSecondary,
        surfaceContainerHigh: Palette.darkDivider,
      );

  // Static color properties for backward compatibility
  static const Color primaryColor = Palette.primaryBlue;
  static const Color primaryBlue = Palette.primaryBlue;
  static const Color secondaryPurple = Color(0xFF8B5CF6);
  static const Color accentGreen = Color(0xFF10B981);

  // Legacy theme getters for main.dart compatibility
  static ThemeData get lightTheme => light;
  static ThemeData get darkTheme => dark;

  // Legacy color getters for backward compatibility
  static Color getBackgroundColor(bool darkMode) => darkMode ? _darkColorScheme.surface : _lightColorScheme.surface;
  static Color getTextColor(bool darkMode) => darkMode ? _darkColorScheme.onSurface : _lightColorScheme.onSurface;
  static Color getCardBackgroundColor(bool darkMode) =>
      darkMode ? _darkColorScheme.surfaceContainer : _lightColorScheme.surfaceContainer;
  static Color getBorderColor(bool darkMode) => darkMode ? _darkColorScheme.outline : _lightColorScheme.outline;
  static Color getSecondaryBorderColor(bool darkMode) =>
      darkMode ? _darkColorScheme.outlineVariant : _lightColorScheme.outlineVariant;
  static Color getMutedTextColor(bool darkMode) =>
      darkMode ? _darkColorScheme.onSurfaceVariant : _lightColorScheme.onSurfaceVariant;
  static Color getSecondaryBackgroundColor(bool darkMode) =>
      darkMode ? _darkColorScheme.surfaceContainerLow : _lightColorScheme.surfaceContainerLow;
  static Color getDividerColor(bool darkMode) =>
      darkMode ? _darkColorScheme.surfaceContainerHigh : _lightColorScheme.surfaceContainerHigh;

  static Color getAppBarBackgroundColor(bool darkMode) => Colors.transparent;
  static Color getSecondaryColor(bool darkMode) => darkMode ? _darkColorScheme.outline : const Color(0xFFE5F1FF);
  static Color getSecondaryTextColor(bool darkMode) => darkMode ? const Color(0xFF97C3FF) : const Color(0xFF1366E1);

  /// Status bar theming based on current theme mode
  static SystemUiOverlayStyle getSystemUiOverlayStyle(bool darkMode) {
    return SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: darkMode ? Brightness.light : Brightness.dark,
      statusBarBrightness: darkMode ? Brightness.dark : Brightness.light,
    );
  }

  static ThemeData get light => _buildTheme(_lightColorScheme);

  static ThemeData get dark => _buildTheme(_darkColorScheme);

  static ThemeData _buildTheme(ColorScheme colorScheme) {
    final isDark = colorScheme.brightness == Brightness.dark;

    return ThemeData(
      colorScheme: colorScheme,
      useMaterial3: true,
      appBarTheme: AppBarTheme(
        backgroundColor: colorScheme.surface,
        foregroundColor: colorScheme.onSurface,
        elevation: Palette.elevationAppBar,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: colorScheme.primary,
          foregroundColor: colorScheme.onPrimary,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Palette.radiusSmall)),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: colorScheme.primary,
          side: BorderSide(color: colorScheme.primary),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Palette.radiusSmall)),
        ),
      ),
      cardTheme: CardThemeData(
        elevation: Palette.elevationCard,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Palette.radiusMedium)),
        color: colorScheme.surfaceContainer,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: colorScheme.surfaceContainer,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Palette.radiusMedium),
          borderSide: BorderSide(color: colorScheme.outline),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Palette.radiusMedium),
          borderSide: BorderSide(color: colorScheme.outline),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Palette.radiusMedium),
          borderSide: BorderSide(color: colorScheme.primary, width: 2),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
      extensions: <ThemeExtension<dynamic>>[AppTextTheme.create(isDark: isDark)],
    );
  }
}

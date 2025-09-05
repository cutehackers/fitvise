import 'package:flutter/material.dart';

/// Design tokens for consistent styling across the app
class Palette {
  Palette._();

  // Brand colors
  static const Color primaryBlue = Color(0xFF377DFF);
  static const Color accentTeal = Color(0xFF00C9A7);

  // Light theme colors
  static const Color lightBackground = Color(0xFFF8F9FA);
  static const Color lightSurface = Colors.white;
  static const Color lightText = Color(0xFF1E2022);
  static const Color lightBorder = Color(0xFFE7EAF0);
  static const Color lightBorderSecondary = Color(0xFFA6A6A6);
  static const Color lightTextMuted = Color(0xFF77838F);
  static const Color lightSurfaceSecondary = Color(0xFFf3f4f6);
  static const Color lightDivider = Color(0xFFe5e7ea);

  // Dark theme colors
  static const Color darkBackground = Color(0xFF1E2022);
  static const Color darkSurface = Color(0xFF2E3336);
  static const Color darkText = Colors.white;
  static const Color darkBorder = Color(0xFF3E444A);
  static const Color darkBorderSecondary = Color(0xFF5A5A5A);
  static const Color darkTextMuted = Color(0xFFADB5BD);
  static const Color darkSurfaceSecondary = Color(0xFF23262A);
  static const Color darkDivider = Color(0xFF34373B);

  // Typography scale
  static const double fontSizeCaption = 12.0;
  static const double fontSizeBodySmall = 13.0;
  static const double fontSizeBodyMedium = 14.0;
  static const double fontSizeBodyLarge = 16.0;
  static const double fontSizeSectionSubtitle = 16.0;
  static const double fontSizeSectionTitle = 18.0;
  static const double fontSizeHeadlineSmall = 20.0;
  static const double fontSizeHeadlineMedium = 22.0;
  static const double fontSizeHeadlineLarge = 24.0;
  static const double fontSizeDisplaySmall = 28.0;
  static const double fontSizeDisplayMedium = 32.0;
  static const double fontSizeDisplayLarge = 36.0;

  // Font weights
  static const FontWeight fontWeightNormal = FontWeight.w400;
  static const FontWeight fontWeightMedium = FontWeight.w500;
  static const FontWeight fontWeightSemiBold = FontWeight.w600;
  static const FontWeight fontWeightBold = FontWeight.w700;

  // Letter spacing
  static const double letterSpacingTight = -0.5;
  static const double letterSpacingNormal = 0.0;
  static const double letterSpacingLoose = 0.2;

  // Border radius
  static const double radiusSmall = 6.0;
  static const double radiusMedium = 8.0;

  // Elevation
  static const double elevationCard = 3.0;
  static const double elevationAppBar = 0.0;
}
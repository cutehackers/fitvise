import 'package:flutter/material.dart';

import 'palette.dart';

class AppTextTheme extends ThemeExtension<AppTextTheme> {
  const AppTextTheme({
    required this.displayLarge,
    required this.displayMedium,
    required this.displaySmall,
    required this.headlineLarge,
    required this.headlineMedium,
    required this.headlineSmall,
    required this.sectionTitle,
    required this.sectionSubtitle,
    required this.bodyLarge,
    required this.bodyMedium,
    required this.bodySmall,
    required this.caption,
  });

  final TextStyle displayLarge;
  final TextStyle displayMedium;
  final TextStyle displaySmall;
  final TextStyle headlineLarge;
  final TextStyle headlineMedium;
  final TextStyle headlineSmall;
  final TextStyle sectionTitle;
  final TextStyle sectionSubtitle;
  final TextStyle bodyLarge;
  final TextStyle bodyMedium;
  final TextStyle bodySmall;
  final TextStyle caption;

  factory AppTextTheme.light() => AppTextTheme.create(isDark: false);

  factory AppTextTheme.dark() => AppTextTheme.create(isDark: true);

  factory AppTextTheme.create({required bool isDark}) {
    final textColor = isDark ? Palette.darkText : Palette.lightText;
    final mutedTextColor = isDark ? Palette.darkTextMuted : Palette.lightTextMuted;
    
    return AppTextTheme(
      displayLarge: _createTextStyle(
        fontSize: Palette.fontSizeDisplayLarge,
        fontWeight: Palette.fontWeightBold,
        color: textColor,
        letterSpacing: Palette.letterSpacingTight,
      ),
      displayMedium: _createTextStyle(
        fontSize: Palette.fontSizeDisplayMedium,
        fontWeight: Palette.fontWeightBold,
        color: textColor,
        letterSpacing: Palette.letterSpacingTight,
      ),
      displaySmall: _createTextStyle(
        fontSize: Palette.fontSizeDisplaySmall,
        fontWeight: Palette.fontWeightBold,
        color: textColor,
        letterSpacing: Palette.letterSpacingTight,
      ),
      headlineLarge: _createTextStyle(
        fontSize: Palette.fontSizeHeadlineLarge,
        fontWeight: Palette.fontWeightBold,
        color: textColor,
      ),
      headlineMedium: _createTextStyle(
        fontSize: Palette.fontSizeHeadlineMedium,
        fontWeight: Palette.fontWeightSemiBold,
        color: textColor,
      ),
      headlineSmall: _createTextStyle(
        fontSize: Palette.fontSizeHeadlineSmall,
        fontWeight: Palette.fontWeightSemiBold,
        color: textColor,
      ),
      sectionTitle: _createTextStyle(
        fontSize: Palette.fontSizeSectionTitle,
        fontWeight: Palette.fontWeightBold,
        color: textColor,
      ),
      sectionSubtitle: _createTextStyle(
        fontSize: Palette.fontSizeSectionSubtitle,
        fontWeight: Palette.fontWeightMedium,
        color: mutedTextColor,
      ),
      bodyLarge: _createTextStyle(
        fontSize: Palette.fontSizeBodyLarge,
        fontWeight: Palette.fontWeightNormal,
        color: textColor,
      ),
      bodyMedium: _createTextStyle(
        fontSize: Palette.fontSizeBodyMedium,
        fontWeight: Palette.fontWeightNormal,
        color: textColor,
      ),
      bodySmall: _createTextStyle(
        fontSize: Palette.fontSizeBodySmall,
        fontWeight: Palette.fontWeightNormal,
        color: mutedTextColor,
      ),
      caption: _createTextStyle(
        fontSize: Palette.fontSizeCaption,
        fontWeight: Palette.fontWeightNormal,
        color: mutedTextColor,
        letterSpacing: Palette.letterSpacingLoose,
      ),
    );
  }

  static TextStyle _createTextStyle({
    required double fontSize,
    required FontWeight fontWeight,
    required Color color,
    double letterSpacing = Palette.letterSpacingNormal,
  }) {
    return TextStyle(
      fontSize: fontSize,
      fontWeight: fontWeight,
      color: color,
      letterSpacing: letterSpacing,
      height: 1.4, // Improved line height for accessibility
    );
  }

  @override
  ThemeExtension<AppTextTheme> copyWith({
    TextStyle? displayLarge,
    TextStyle? displayMedium,
    TextStyle? displaySmall,
    TextStyle? headlineLarge,
    TextStyle? headlineMedium,
    TextStyle? headlineSmall,
    TextStyle? sectionTitle,
    TextStyle? sectionSubtitle,
    TextStyle? bodyLarge,
    TextStyle? bodyMedium,
    TextStyle? bodySmall,
    TextStyle? caption,
  }) {
    return AppTextTheme(
      displayLarge: displayLarge ?? this.displayLarge,
      displayMedium: displayMedium ?? this.displayMedium,
      displaySmall: displaySmall ?? this.displaySmall,
      headlineLarge: headlineLarge ?? this.headlineLarge,
      headlineMedium: headlineMedium ?? this.headlineMedium,
      headlineSmall: headlineSmall ?? this.headlineSmall,
      sectionTitle: sectionTitle ?? this.sectionTitle,
      sectionSubtitle: sectionSubtitle ?? this.sectionSubtitle,
      bodyLarge: bodyLarge ?? this.bodyLarge,
      bodyMedium: bodyMedium ?? this.bodyMedium,
      bodySmall: bodySmall ?? this.bodySmall,
      caption: caption ?? this.caption,
    );
  }

  @override
  ThemeExtension<AppTextTheme> lerp(covariant ThemeExtension<AppTextTheme>? other, double t) {
    if (other == null || other is! AppTextTheme) {
      return this;
    }
    return AppTextTheme(
      displayLarge: TextStyle.lerp(displayLarge, other.displayLarge, t)!,
      displayMedium: TextStyle.lerp(displayMedium, other.displayMedium, t)!,
      displaySmall: TextStyle.lerp(displaySmall, other.displaySmall, t)!,
      headlineLarge: TextStyle.lerp(headlineLarge, other.headlineLarge, t)!,
      headlineMedium: TextStyle.lerp(headlineMedium, other.headlineMedium, t)!,
      headlineSmall: TextStyle.lerp(headlineSmall, other.headlineSmall, t)!,
      sectionTitle: TextStyle.lerp(sectionTitle, other.sectionTitle, t)!,
      sectionSubtitle: TextStyle.lerp(sectionSubtitle, other.sectionSubtitle, t)!,
      bodyLarge: TextStyle.lerp(bodyLarge, other.bodyLarge, t)!,
      bodyMedium: TextStyle.lerp(bodyMedium, other.bodyMedium, t)!,
      bodySmall: TextStyle.lerp(bodySmall, other.bodySmall, t)!,
      caption: TextStyle.lerp(caption, other.caption, t)!,
    );
  }
}
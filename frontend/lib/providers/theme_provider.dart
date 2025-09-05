import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

// Provider
final themeModeProvider = StateNotifierProvider<ThemeModeNotifier, ThemeMode>((ref) => ThemeModeNotifier());

// ThemeMode 상태 관리
class ThemeModeNotifier extends StateNotifier<ThemeMode> {
  late SharedPreferences _prefs;

  ThemeModeNotifier() : super(ThemeMode.system) {
    load();
  }

  bool get isDarkMode => state == ThemeMode.dark;

  Future<void> load() async {
    _prefs = await SharedPreferences.getInstance();

    // Check saved preference first, then system preference
    final theme = _prefs.getString('prefs_theme');
    if (theme != null) {
      state = switch (theme) {
        'system' => ThemeMode.system,
        'light' => ThemeMode.light,
        'dark' => ThemeMode.dark,
        _ => ThemeMode.system,
      };
    } else {
      final brightness = WidgetsBinding.instance.platformDispatcher.platformBrightness;
      state = brightness == Brightness.dark ? ThemeMode.dark : ThemeMode.system;
    }
  }

  // 테마 모드 변경 메소드
  void changeThemeMode(ThemeMode mode) => state = mode;

  // 테마 모드 토글 메소드
  void toggleThemeMode() {
    switch (state) {
      case ThemeMode.system:
        print('Theme> System -> Light');
        _prefs.setString('prefs_theme', 'light');
        state = ThemeMode.light;
        break;
      case ThemeMode.light:
        print('Theme> Light -> Dark');
        _prefs.setString('prefs_theme', 'light');
        state = ThemeMode.dark;
        break;
      case ThemeMode.dark:
        print('Theme> Dark -> Light');
        _prefs.setString('prefs_theme', 'light');
        state = ThemeMode.light;
        break;
    }
  }
}

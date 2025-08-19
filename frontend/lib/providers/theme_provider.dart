import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ThemeProvider extends ChangeNotifier {
  bool _isDarkMode = false;
  SharedPreferences? _prefs;

  bool get isDarkMode => _isDarkMode;
  String get theme => _isDarkMode ? 'dark' : 'light';

  ThemeProvider() {
    _loadTheme();
  }

  Future<void> _loadTheme() async {
    _prefs = await SharedPreferences.getInstance();
    
    // Check saved preference first, then system preference
    final savedTheme = _prefs?.getString('fitvise_theme');
    if (savedTheme != null) {
      _isDarkMode = savedTheme == 'dark';
    } else {
      // Use system preference if no saved preference
      final brightness = WidgetsBinding.instance.platformDispatcher.platformBrightness;
      _isDarkMode = brightness == Brightness.dark;
    }
    
    notifyListeners();
  }

  Future<void> toggleTheme() async {
    _isDarkMode = !_isDarkMode;
    
    // Save to preferences
    await _prefs?.setString('fitvise_theme', _isDarkMode ? 'dark' : 'light');
    
    notifyListeners();
  }

  Future<void> setTheme(bool isDark) async {
    if (_isDarkMode != isDark) {
      _isDarkMode = isDark;
      await _prefs?.setString('fitvise_theme', _isDarkMode ? 'dark' : 'light');
      notifyListeners();
    }
  }
}
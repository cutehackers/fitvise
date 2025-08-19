import 'package:flutter/foundation.dart';

/// Application configuration class for managing environment-specific settings
class AppConfig {
  // Private constructor to prevent instantiation
  AppConfig._();

  /// Base URL for the API
  static String get baseUrl {
    // Check for environment variable first
    const envBaseUrl = String.fromEnvironment('API_BASE_URL');
    if (envBaseUrl.isNotEmpty) {
      return envBaseUrl;
    }
    
    // Default URLs based on build mode
    if (kDebugMode) {
      // Local development URL
      return 'http://localhost:8000';
    } else {
      // Production URL
      return 'https://api.fitvise.com';
    }
  }

  /// API version prefix
  static const String apiVersion = '/api/v1';

  /// Full API base URL with version
  static String get apiBaseUrl => '$baseUrl$apiVersion';

  /// Timeout configurations
  static const Duration connectTimeout = Duration(seconds: 30);
  static const Duration receiveTimeout = Duration(seconds: 30);
  static const Duration sendTimeout = Duration(seconds: 30);

  /// Feature flags
  static const bool enableApiLogging = kDebugMode;
  static const bool enableDetailedErrors = kDebugMode;

  /// App information
  static const String appName = 'Fitvise';
  static const String appVersion = '1.0.0';
}
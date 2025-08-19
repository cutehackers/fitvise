import 'package:dio/dio.dart' hide Headers;
import 'package:flutter/foundation.dart';
import 'package:retrofit/retrofit.dart';

import '../config/app_config.dart';

part 'agent_api.g.dart';

final dio = ApiClient.createDio();

final agentApi = ApiClient.createAgentApi(dio: dio);

@RestApi()
abstract class AgentApi {
  factory AgentApi(Dio dio, {String baseUrl}) = _AgentApi;

  // @POST("/api/v1/fitvise/chat")
  // @Headers(<String, dynamic>{'Accept': 'application/x-ndjson'})
  // @DioResponseType(ResponseType.stream)
  // Future<Response<ResponseBody>> chat(@Body() ChatRequest request);
}

/// Create a configured Dio instance for the AgentApi
class ApiClient {
  static Dio createDio() {
    final dio = Dio();

    // Configure base options
    dio.options = BaseOptions(
      baseUrl: AppConfig.baseUrl,
      connectTimeout: AppConfig.connectTimeout,
      receiveTimeout: AppConfig.receiveTimeout,
      sendTimeout: AppConfig.sendTimeout,
      headers: {'Content-Type': 'application/json'},
    );

    // Add interceptors for logging in debug mode
    if (AppConfig.enableApiLogging) {
      dio.interceptors.add(
        LogInterceptor(
          requestBody: true,
          responseBody: true,
          error: true,
          logPrint: (object) {
            // Use debugPrint for development logging
            debugPrint('[API] $object');
          },
        ),
      );
    }

    // Add error handling interceptor
    dio.interceptors.add(
      InterceptorsWrapper(
        onError: (error, handler) {
          // Handle common errors here
          switch (error.response?.statusCode) {
            case 401:
              // Handle unauthorized
              debugPrint('[API] Unauthorized access - check API credentials');
              break;
            case 404:
              // Handle not found
              debugPrint('[API] Resource not found - check API endpoint');
              break;
            case 429:
              // Handle rate limiting
              debugPrint('[API] Rate limit exceeded - please try again later');
              break;
            case 500:
              // Handle server error
              debugPrint('[API] Server error - backend is experiencing issues');
              break;
            case 503:
              // Handle service unavailable
              debugPrint('[API] Service temporarily unavailable');
              break;
            default:
              // Handle other errors
              debugPrint('[API] Request failed: ${error.message}');
          }
          handler.next(error);
        },
      ),
    );

    return dio;
  }

  static AgentApi createAgentApi({required Dio dio, String? baseUrl}) {
    return AgentApi(dio, baseUrl: baseUrl ?? AppConfig.baseUrl);
  }
}

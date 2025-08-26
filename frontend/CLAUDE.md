# CLAUDE.md

Fitvise is an AI fitness chat application that provides personalized workout guidance and health insights through conversational interface.

## Application Overview

### Architecture Pattern
Flutter AI fitness chat application using Clean Architecture principles with Provider state management.

### Core Features
- **AI Chat Interface**: Conversational fitness guidance with welcome prompts for workout, nutrition, progress, education, goals, and wellness
- **Real-time Communication**: Typing indicators, message editing, quick replies, and voice input support
- **Activity Tracking**: Health platform integration with workout history
- **Photo Editing**: Advanced photo editing with activity overlays using `pro_image_editor`
- **Social Sharing**: Share workout achievements via `share_plus`

### Directory Structure
```
lib/
├── core/                    # Shared utilities, permissions, theming
├── providers/               # Global state management providers
├── startup/                 # App initialization logic
├── ui/
│   ├── feature/{name}/      # Feature-based organization
│   │   ├── data/           # Data layer (datasources, models, repositories, services)
│   │   ├── domain/         # Business logic (entities, use cases)
│   │   └── widget/         # Feature-specific UI widgets
│   └── widget/             # Shared UI components
├── extension/              # Dart extensions
├── models/                 # Chat-specific data models
├── screens/                # Application screens
├── widgets/                # Chat UI components
└── theme/                  # App theming
```

## State Management

### Provider Architecture
Uses Provider pattern with two main providers:
- `ChatProvider` - Chat messages, typing states, AI responses, message history, editing functionality
- `ThemeProvider` - Light/dark theme switching with SharedPreferences persistence

### Provider Naming Conventions
| Provider Type | Provider Name | Notifier Class Name | State Type |
|---------------|---------------|---------------------|------------|
| `StateProvider` | `activityListProvider` | — | `List<Activity>` |
| `StateNotifierProvider` | `activityListProvider` | `ActivityListNotifier` | `List<Activity>` |
| `NotifierProvider` | `activityListProvider` | `ActivityListNotifier` | `List<Activity>` |
| `AsyncNotifierProvider` | `activityListProvider` | `ActivityListNotifier` | `AsyncValue<List<Activity>>` |
| `FutureProvider` | `activityListFutureProvider` | — | `AsyncValue<List<Activity>>` |
| `StreamProvider` | `activityListStreamProvider` | — | `AsyncValue<List<Activity>>` |

## Code Standards

### Style Guidelines
- Flutter lints + custom_lint, riverpod_lint, freezed_lint
- Implicit casts disabled (strong typing)
- Single quotes preferred
- Trailing commas required
- Generated files excluded from analysis (*.g.dart, *.freezed.dart)

### UI Component Hierarchy
```
Screen → View → Layout → Group → (Atom)
```
- Atomic components have no suffix
- Follow existing naming patterns in `lib/ui/widget/`
- Create seperate widget component class rahter then using `Widget _build...` inner method.

### Code Generation
- **Freezed & JSON**: Generates for `lib/**/domain/entity/**.dart` - immutable data classes with JSON serialization
- **Drift Database**: SQLite 3.38+ with FTS5 and JSON1 modules
  - Database file: `lib/ui/feature/activity/data/datasources/local/database.dart`
  - Schema directory: `schemas/`
  - Test directory: `test/drift/`

## Testing Strategy

### Architecture Testing Rules
| Layer | Coverage Target | Focus Areas |
|-------|-----------------|-------------|
| **Domain** | 95% | Business logic isolation, entity validation, error handling, immutability |
| **Data** | 90% | Repository implementation, JSON serialization, caching, network errors |
| **Presentation** | 85% | Provider testing, state management, widget testing, navigation |

### Implementation Guidelines

**Mock Generation**:
```dart
@GenerateMocks([ApiService, Repository, DataSource])
void main() {}
// Run: flutter packages pub run build_runner build
```

**Provider Testing**:
```dart
test('should return data when repository succeeds', () async {
  final container = ProviderContainer(
    overrides: [repositoryProvider.overrideWithValue(mockRepository)],
  );
  when(mockRepository.getData()).thenAnswer((_) async => testData);
  final result = await container.read(dataProvider.future);
  expect(result, testData);
});
```

**Widget Testing with Riverpod**:
```dart
testWidgets('should display data when loaded', (tester) async {
  await tester.pumpWidget(
    ProviderScope(
      overrides: [dataProvider.overrideWith((ref) => AsyncValue.data(testData))],
      child: MyWidget(),
    ),
  );
  expect(find.text(testData.title), findsOneWidget);
});
```

### Naming & Performance Standards
- **Files**: `{filename}_test.dart`, `mock_{service_name}.dart`, `test_{entity_name}.dart`
- **Performance**: Unit tests <50ms, widget tests <200ms
- **Strategy**: Mock external dependencies, use `when()`/`verify()`, avoid real network calls

### Commands
```bash
flutter test                    # Run all tests
flutter test --coverage        # Run with coverage
flutter test path/to/test.dart  # Run specific test
```

## Key Integrations

### Data
- **Database**: Drift with compile-time SQL validation, DAO pattern, platform-specific connections
- **API Integration**: Dio + Retrofit for type-safe API calls to `/api/v1/prompt` endpoint
- **Error Handling**: Comprehensive error handling with user-friendly messages for network/timeout/auth failures

### AI Backend System
- **API Architecture**: `AgentApi` class with Retrofit annotations, @freezed models (`PromptRequest`/`PromptResponse`)
- **Communication**: Real-time chat with typing indicators, message editing, quick replies
- **Response Processing**: Timeout/retry mechanisms, graceful degradation for production use

## Environment & Development

### API Configuration
- **Local**: `http://localhost:8000` (debug mode default)
- **Development**: `https://dev-api.fitvise.com` (release mode)
- **Production**: `https://api.fitvise.com` (release mode)
- **Custom**: `--dart-define=API_BASE_URL=your_url`
- **Config**: Managed in `lib/config/app_config.dart` with automatic build mode selection

### Commands
```bash
# Application
flutter run                                        # Debug mode
flutter run --release                              # Production mode
flutter run --dart-define=API_BASE_URL=custom_url  # Custom API
flutter build apk|ios|web                          # Build platforms

# Development
flutter pub get|upgrade        # Dependencies
flutter clean                  # Clean build
flutter analyze                # Static analysis
flutter format .              # Code formatting

# Code Generation
flutter packages pub run build_runner build       # Generate code
```

### Dependencies
**Runtime**:
- `provider: ^6.1.1` - State management
- `shared_preferences: ^2.2.2` - Theme persistence  
- `dio: ^5.4.0` + `retrofit: ^4.1.0` - API client
- `health` + `permission_handler` - Health integration
- `speech_to_text: ^6.6.0` - Voice input
- `freezed_annotation: ^3.1.0` + `json_annotation: ^4.8.1` - Models

**Development**:
- `build_runner: ^2.4.7` - Code generation
- `freezed: ^3.2.0` + `json_serializable: ^6.7.1` - Model generation
- `retrofit_generator: ^10.0.0` - API generation

### Platform Support
- **Mobile**: Android (AndroidManifest.xml permissions), iOS (Info.plist configurations)
- **Desktop**: macOS, Windows (basic configuration)
- **Web**: manifest.json and icons

### Development Notes
- Run code generation after modifying entities/database schema
- Test structure mirrors main code organization
- Follow clean architecture: Data → Domain → UI
- Generated files excluded from version control
- Basic widget test exists expecting "Fitvise AI" text
# Chat Widget Components

A scalable and reusable Flutter chat widget library built with modular components for the Fitvise app.

## Overview

This library provides a comprehensive set of modular chat components that can be used individually or combined to create sophisticated chat interfaces. The components are designed for maximum reusability, customization, and performance.

## Architecture

### Modular Design
- **Component-based**: Each widget has a single responsibility
- **Composable**: Components can be combined in different ways
- **Customizable**: Extensive configuration options
- **Reusable**: Works across different chat scenarios

### Component Categories

#### Core Message Components
- `AnimatedTextMessage` - Word-by-word streaming text animation
- `MessageBubble` - Customizable message bubbles
- `MessageAttachment` - File/media attachment display

#### Input & Interaction
- `ChatInput` - Modern input field with attachments and voice
- `ChatLoadingWidget` - Various loading animations and indicators

#### UI & Effects
- `GlassmorphicContainer` - Translucent UI effects and containers

## Components

### AnimatedTextMessage

Displays text with word-by-word streaming animation, perfect for AI responses.

```dart
AnimatedTextMessage(
  text: "This text appears word by word!",
  wordDelay: Duration(milliseconds: 100),
  showCursor: true,
  onComplete: () => print("Animation complete"),
)
```

**Features:**
- Word-by-word streaming animation
- Customizable timing and styling
- Typing cursor support
- Performance optimized for long texts

### MessageBubble

Highly customizable message bubble for chat interfaces.

```dart
MessageBubble(
  text: "Hello, world!",
  isUser: true,
  timestamp: DateTime.now(),
  senderName: "John Doe",
  config: MessageBubbleConfig(
    borderRadius: 20,
    showTimestamp: true,
    gradient: LinearGradient(colors: [Colors.blue, Colors.purple]),
  ),
)
```

**Features:**
- User and AI message support
- Customizable appearance themes
- Avatar and timestamp support
- Action buttons (copy, edit, etc.)
- Animation support
- Accessibility compliant

### ChatInput

Modern input field with comprehensive features.

```dart
ChatInput(
  controller: textController,
  onSend: (text) => sendMessage(text),
  config: ChatInputConfig(
    enableVoiceInput: true,
    enableAttachments: true,
    maxLength: 2000,
    attachmentOptions: [
      AttachmentOption(
        icon: Icons.image,
        tooltip: "Upload image",
        type: "image",
      ),
    ],
  ),
)
```

**Features:**
- Glassmorphic design with gradients
- Voice input support
- File attachment options
- Character count and status indicators
- Auto-expanding text field
- Keyboard shortcuts support

### ChatLoadingWidget

Collection of loading animations for different states.

```dart
// Typing indicator
ChatLoadingWidget(
  type: LoadingType.typing,
  message: "AI is thinking...",
)

// Different animation types
ChatLoadingWidget(type: LoadingType.pulse)
ChatLoadingWidget(type: LoadingType.wave)
ChatLoadingWidget(type: LoadingType.dots)
ChatLoadingWidget(type: LoadingType.spinner)
```

**Loading Types:**
- `typing` - Bouncing dots for typing indication
- `pulse` - Pulsing circle animation
- `wave` - Wave-like bars animation
- `dots` - Bouncing dots animation
- `spinner` - Rotating spinner

### MessageAttachment

Versatile attachment display widget.

```dart
MessageAttachmentWidget(
  attachment: MessageAttachment(
    id: "1",
    name: "workout.jpg",
    type: "image",
    url: "https://example.com/image.jpg",
    size: 1024000,
  ),
  onTap: () => viewAttachment(),
  onDownload: () => downloadAttachment(),
  showActions: true,
)
```

**Features:**
- Image, video, audio, and document support
- Thumbnail generation
- Progress indicators for uploads/downloads
- Download and view actions
- Error handling and retry

### GlassmorphicContainer

Reusable translucent container with blur effects.

```dart
// Predefined styles
GlassmorphicContainer.light(
  child: Text("Light glass effect"),
)

GlassmorphicContainer.dark(
  child: Text("Dark glass effect"),
)

GlassmorphicContainer.gradient(
  gradient: LinearGradient(colors: [Colors.blue, Colors.purple]),
  child: Text("Gradient glass effect"),
)

// Custom configuration
GlassmorphicContainer(
  config: GlassmorphicConfig(
    blur: 15.0,
    opacity: 0.1,
    borderRadius: BorderRadius.circular(20),
  ),
  child: YourWidget(),
)
```

**Features:**
- Multiple glass styles (light, dark, gradient, frosted)
- Customizable blur and opacity
- Border and shadow effects
- Extension methods for easy usage

## Usage Examples

### Basic Chat Interface

```dart
class SimpleChatPage extends StatefulWidget {
  @override
  State<SimpleChatPage> createState() => _SimpleChatPageState();
}

class _SimpleChatPageState extends State<SimpleChatPage> {
  final TextEditingController _controller = TextEditingController();
  final List<String> _messages = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Messages
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                return MessageBubble(
                  text: _messages[index],
                  isUser: index % 2 == 0,
                  timestamp: DateTime.now(),
                );
              },
            ),
          ),
          
          // Input
          SimpleChatInput(
            controller: _controller,
            onSend: (text) {
              setState(() {
                _messages.add(text);
              });
              _controller.clear();
            },
          ),
        ],
      ),
    );
  }
}
```

### Advanced Chat with Streaming

```dart
class AdvancedChatPage extends StatefulWidget {
  @override
  State<AdvancedChatPage> createState() => _AdvancedChatPageState();
}

class _AdvancedChatPageState extends State<AdvancedChatPage> {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, dynamic>> _messages = [];
  bool _isTyping = false;

  void _sendMessage(String text) {
    setState(() {
      _messages.add({
        'text': text,
        'isUser': true,
        'timestamp': DateTime.now(),
      });
      _isTyping = true;
    });

    // Simulate AI response with streaming
    Future.delayed(Duration(seconds: 1), () {
      setState(() {
        _messages.add({
          'text': 'AI response with streaming animation',
          'isUser': false,
          'timestamp': DateTime.now(),
          'streaming': true,
        });
        _isTyping = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Messages
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                return MessageBubble(
                  text: message['text'],
                  isUser: message['isUser'],
                  timestamp: message['timestamp'],
                  content: message['streaming'] == true
                      ? AnimatedTextMessage(
                          text: message['text'],
                          wordDelay: Duration(milliseconds: 100),
                        )
                      : null,
                );
              },
            ),
          ),
          
          // Typing indicator
          if (_isTyping)
            TypingIndicator(message: "AI is typing..."),
          
          // Input
          ChatInput(
            controller: _controller,
            onSend: _sendMessage,
            config: ChatInputConfig(
              enableVoiceInput: true,
              enableAttachments: true,
            ),
          ),
        ],
      ),
    );
  }
}
```

## Configuration

### Theming

All components respect the app's theme and provide additional customization options:

```dart
// Custom message bubble theme
final customBubbleConfig = MessageBubbleConfig(
  backgroundColor: Colors.blue.shade100,
  borderRadius: 20,
  padding: EdgeInsets.all(16),
  textStyle: TextStyle(fontSize: 16, color: Colors.blue.shade900),
);

// Custom input theme
final customInputConfig = ChatInputConfig(
  borderColor: Colors.blue,
  gradient: LinearGradient(
    colors: [Colors.blue.withOpacity(0.1), Colors.purple.withOpacity(0.1)],
  ),
  hintText: "Type your message...",
);
```

### Performance Optimization

- Components use `const` constructors where possible
- Animations are optimized for 60fps performance
- Large lists use efficient rendering techniques
- Memory usage is minimized through proper disposal

## Migration from AiChatWidget

The original `AiChatWidget` has been refactored to use these modular components while maintaining backward compatibility:

```dart
// Old approach - single monolithic widget
AiChatWidget(
  sessionId: "session-123",
  showWelcomePrompts: true,
)

// New approach - using modular components
AiChatWidget(
  sessionId: "session-123", 
  showWelcomePrompts: true,
  // Now internally uses MessageBubble, ChatInput, etc.
)

// Or build custom chat with individual components
Column(
  children: [
    Expanded(
      child: ListView(
        children: messages.map((message) => MessageBubble(...)).toList(),
      ),
    ),
    ChatInput(...),
  ],
)
```

## Best Practices

### Performance
- Use `const` constructors when possible
- Implement proper `dispose()` methods for controllers
- Use `RepaintBoundary` for complex animations
- Consider `ListView.builder` for large message lists

### Accessibility
- All components include semantic labels
- Support for screen readers
- Keyboard navigation support
- High contrast mode compatibility

### Testing
- Each component is unit testable
- Widget tests cover common use cases
- Golden tests for visual regression
- Performance benchmarks included

## Dependencies

```yaml
dependencies:
  flutter: ^3.0.0
  provider: ^6.0.0
```

## License

Part of the Fitvise Flutter application.
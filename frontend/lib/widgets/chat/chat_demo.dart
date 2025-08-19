import 'package:flutter/material.dart';
import 'index.dart';
import '../../theme/app_theme.dart';

/// Demo page showcasing the modular chat components
/// 
/// This demonstrates how to use the individual chat components
/// for building custom chat interfaces or different chat scenarios.
class ChatComponentsDemo extends StatefulWidget {
  const ChatComponentsDemo({super.key});

  @override
  State<ChatComponentsDemo> createState() => _ChatComponentsDemoState();
}

class _ChatComponentsDemoState extends State<ChatComponentsDemo> {
  final TextEditingController _controller = TextEditingController();
  final List<String> _messages = [];
  bool _isTyping = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _addMessage(String text) {
    setState(() {
      _messages.add(text);
      _isTyping = true;
    });

    // Simulate AI response
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) {
        setState(() {
          _messages.add('This is a demo AI response to: $text');
          _isTyping = false;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chat Components Demo'),
        backgroundColor: AppTheme.primaryBlue,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Demo sections
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildSection(
                    'Message Bubbles',
                    Column(
                      children: [
                        MessageBubble(
                          text: 'Hello! This is a user message.',
                          isUser: true,
                          timestamp: DateTime.now(),
                        ),
                        const SizedBox(height: 8),
                        MessageBubble(
                          text: 'And this is an AI response with a longer text that demonstrates how the bubble adapts to different content lengths.',
                          isUser: false,
                          timestamp: DateTime.now(),
                          senderName: 'Fitvise AI',
                        ),
                        const SizedBox(height: 8),
                        SystemMessageBubble(
                          text: 'This is a system message',
                          icon: Icons.info,
                        ),
                      ],
                    ),
                  ),
                  
                  _buildSection(
                    'Animated Text',
                    AnimatedTextMessage(
                      text: 'This text appears word by word with smooth animations!',
                      style: const TextStyle(fontSize: 16),
                      wordDelay: const Duration(milliseconds: 200),
                      showCursor: true,
                    ),
                  ),
                  
                  _buildSection(
                    'Loading Indicators',
                    Column(
                      children: [
                        const TypingIndicator(
                          message: 'AI is thinking...',
                        ),
                        const SizedBox(height: 16),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: const [
                            ChatLoadingWidget(type: LoadingType.typing),
                            ChatLoadingWidget(type: LoadingType.pulse),
                            ChatLoadingWidget(type: LoadingType.wave),
                            ChatLoadingWidget(type: LoadingType.dots),
                            ChatLoadingWidget(type: LoadingType.spinner),
                          ],
                        ),
                      ],
                    ),
                  ),
                  
                  _buildSection(
                    'Glassmorphic Effects',
                    Column(
                      children: [
                        GlassmorphicContainer.light(
                          padding: const EdgeInsets.all(16),
                          child: const Text('Light glassmorphic container'),
                        ),
                        const SizedBox(height: 8),
                        GlassmorphicContainer.dark(
                          padding: const EdgeInsets.all(16),
                          child: const Text('Dark glassmorphic container', style: TextStyle(color: Colors.white)),
                        ),
                        const SizedBox(height: 8),
                        GlassmorphicContainer.gradient(
                          gradient: const LinearGradient(
                            colors: [AppTheme.primaryBlue, AppTheme.secondaryPurple],
                          ),
                          padding: const EdgeInsets.all(16),
                          child: const Text('Gradient glassmorphic container', style: TextStyle(color: Colors.white)),
                        ),
                      ],
                    ),
                  ),
                  
                  _buildSection(
                    'Live Chat Demo',
                    Column(
                      children: [
                        Container(
                          height: 200,
                          decoration: BoxDecoration(
                            border: Border.all(color: Colors.grey.shade300),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Column(
                            children: [
                              Expanded(
                                child: ListView.builder(
                                  padding: const EdgeInsets.all(8),
                                  itemCount: _messages.length,
                                  itemBuilder: (context, index) {
                                    final isUser = index % 2 == 0;
                                    return MessageBubble(
                                      text: _messages[index],
                                      isUser: isUser,
                                      timestamp: DateTime.now(),
                                      senderName: isUser ? null : 'Demo AI',
                                      config: const MessageBubbleConfig(
                                        maxWidth: 0.8,
                                        showTimestamp: false,
                                      ),
                                    );
                                  },
                                ),
                              ),
                              if (_isTyping)
                                const TypingIndicator(
                                  message: 'Demo AI is typing...',
                                ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 8),
                        SimpleChatInput(
                          controller: _controller,
                          hintText: 'Try the demo chat...',
                          onSend: (text) {
                            _addMessage(text);
                            _controller.clear();
                          },
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSection(String title, Widget content) {
    return Container(
      margin: const EdgeInsets.only(bottom: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: AppTheme.primaryBlue,
            ),
          ),
          const SizedBox(height: 12),
          content,
        ],
      ),
    );
  }
}

/// Example of a custom chat widget using the modular components
class CustomChatWidget extends StatefulWidget {
  final String title;
  final Color primaryColor;
  final bool enableAttachments;
  final bool enableVoice;

  const CustomChatWidget({
    super.key,
    this.title = 'Custom Chat',
    this.primaryColor = AppTheme.primaryBlue,
    this.enableAttachments = true,
    this.enableVoice = true,
  });

  @override
  State<CustomChatWidget> createState() => _CustomChatWidgetState();
}

class _CustomChatWidgetState extends State<CustomChatWidget> {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, dynamic>> _messages = [];
  bool _isTyping = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _sendMessage(String text) {
    setState(() {
      _messages.add({
        'text': text,
        'isUser': true,
        'timestamp': DateTime.now(),
      });
      _isTyping = true;
    });

    // Simulate response
    Future.delayed(const Duration(milliseconds: 1500), () {
      if (mounted) {
        setState(() {
          _messages.add({
            'text': 'Custom AI response: $text',
            'isUser': false,
            'timestamp': DateTime.now(),
          });
          _isTyping = false;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Header
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: widget.primaryColor,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
          ),
          child: Row(
            children: [
              Icon(Icons.chat, color: Colors.white),
              const SizedBox(width: 8),
              Text(
                widget.title,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
        
        // Messages
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: Colors.grey.shade50,
              border: Border.all(color: Colors.grey.shade300),
            ),
            child: Column(
              children: [
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) {
                      final message = _messages[index];
                      return MessageBubble(
                        text: message['text'],
                        isUser: message['isUser'],
                        timestamp: message['timestamp'],
                        senderName: message['isUser'] ? null : widget.title,
                        config: MessageBubbleConfig(
                          gradient: message['isUser'] 
                              ? LinearGradient(
                                  colors: [widget.primaryColor, widget.primaryColor.withValues(alpha: 0.8)],
                                )
                              : null,
                        ),
                      );
                    },
                  ),
                ),
                
                if (_isTyping)
                  TypingIndicator(
                    message: '${widget.title} is typing...',
                    color: widget.primaryColor,
                  ),
              ],
            ),
          ),
        ),
        
        // Input
        ChatInput(
          controller: _controller,
          onSend: (text) {
            _sendMessage(text);
            _controller.clear();
          },
          config: ChatInputConfig(
            enableAttachments: widget.enableAttachments,
            enableVoiceInput: widget.enableVoice,
            borderColor: widget.primaryColor,
            gradient: LinearGradient(
              colors: [
                widget.primaryColor.withValues(alpha: 0.1),
                widget.primaryColor.withValues(alpha: 0.05),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
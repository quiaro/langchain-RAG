import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  IconButton,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import api from '../utils/api';
import Message from './Message';

const ChatInterface = ({ sessionId, fileName, status, resetSession }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Cleanup function for in-progress requests
  useEffect(() => {
    return () => {
      // Abort any in-progress fetch when component unmounts
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Initial welcome message
  useEffect(() => {
    if (status === 'ready' && messages.length === 0) {
      setMessages([
        {
          role: 'system',
          content: `File "${fileName}" has been processed. You can now ask questions about its content.`,
        },
      ]);
    }
  }, [status, fileName, messages.length]);

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (!input.trim() || isLoading) return;

    // Abort any previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Add user message to chat
    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError('');

    // Create an AbortController to handle cleanup
    const controller = new AbortController();
    const signal = controller.signal;
    abortControllerRef.current = controller;

    // Use a ref to track the accumulated response
    const responseTextRef = { current: '' };
    const tempId = Date.now().toString();

    try {
      // Prepare assistant response placeholder
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '', id: tempId, loading: true },
      ]);

      // Using fetch API with proper headers for streaming
      const response = await fetch(`${api.defaults.baseURL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({
          session_id: sessionId,
          query: userMessage.content,
        }),
        signal, // Add the abort signal
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done, value;

      while (!done) {
        ({ done, value } = await reader.read());

        // Check if we need to abort
        if (signal.aborted || done) {
          break;
        }

        // Decode the chunk and append to accumulated response
        const chunk = decoder.decode(value, { stream: true });
        responseTextRef.current += chunk;

        // Update the message with the current response text
        setMessages((prev) => {
          const updatedMessages = [...prev];
          const assistantMsgIndex = updatedMessages.findIndex(
            (msg) => msg.id === tempId
          );

          if (assistantMsgIndex !== -1) {
            updatedMessages[assistantMsgIndex] = {
              ...updatedMessages[assistantMsgIndex],
              content: responseTextRef.current,
              loading: true,
            };
          }

          return updatedMessages;
        });
      }

      // Only finalize if not aborted
      if (!signal.aborted) {
        // Mark the message as complete
        setMessages((prev) => {
          const updatedMessages = [...prev];
          const assistantMsgIndex = updatedMessages.findIndex(
            (msg) => msg.id === tempId
          );

          if (assistantMsgIndex !== -1) {
            updatedMessages[assistantMsgIndex] = {
              role: 'assistant',
              content: responseTextRef.current,
              id: tempId,
              loading: false,
            };
          }

          return updatedMessages;
        });
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Fetch aborted');
        return;
      }

      console.error('Query error:', error);
      setError(error.message || 'Failed to get a response');

      // Remove the loading message
      setMessages((prev) => prev.filter((msg) => !(msg.id && msg.loading)));
    } finally {
      if (!signal.aborted) {
        setIsLoading(false);
        // Clear the abortControllerRef if this request is complete and not aborted
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      }
    }
  };

  const handleDeleteSession = async () => {
    try {
      await api.delete(`/session/${sessionId}`);
      resetSession();
    } catch (error) {
      console.error('Error deleting session:', error);
      setError('Failed to reset chat');
    }
  };

  return (
    <Paper
      elevation={3}
      sx={{ height: '80vh', display: 'flex', flexDirection: 'column' }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Box>
          <Typography variant="h6">Chat with {fileName}</Typography>
          <Typography variant="body2" color="text.secondary">
            Ask questions about the content of your file
          </Typography>
        </Box>
        <Button
          variant="outlined"
          color="error"
          startIcon={<DeleteIcon />}
          onClick={handleDeleteSession}
        >
          Reset
        </Button>
      </Box>

      {/* Messages Container */}
      <Box
        sx={{
          p: 2,
          flexGrow: 1,
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {messages.map((message, index) => (
          <Message key={message.id || index} message={message} />
        ))}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Box */}
      <Box sx={{ p: 2, borderTop: '1px solid rgba(255, 255, 255, 0.12)' }}>
        <form onSubmit={handleSendMessage}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Type your question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading || status !== 'ready'}
              sx={{ mr: 1 }}
            />
            <IconButton
              color="primary"
              type="submit"
              disabled={isLoading || !input.trim() || status !== 'ready'}
            >
              {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
            </IconButton>
          </Box>
        </form>
      </Box>
    </Paper>
  );
};

export default ChatInterface;

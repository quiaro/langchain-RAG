import React from 'react';
import { Box, Paper, Typography, CircularProgress } from '@mui/material';

const Message = ({ message }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  // Styling based on message role
  const messageStyle = {
    maxWidth: '80%',
    alignSelf: isUser ? 'flex-end' : 'flex-start',
    backgroundColor: isSystem
      ? 'rgba(255, 255, 255, 0.08)'
      : isUser
      ? '#1976d2'
      : 'rgba(255, 255, 255, 0.12)',
    color: isUser ? '#fff' : 'inherit',
    borderRadius: '12px',
    p: 2,
  };

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      <Paper elevation={1} sx={messageStyle}>
        {message.content === '' ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={20} />
            <Typography variant="body1">Thinking...</Typography>
          </Box>
        ) : (
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
            {message.content}
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

export default Message;

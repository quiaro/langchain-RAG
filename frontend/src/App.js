import React, { useState } from 'react';
import {
  Container,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from '@mui/material';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';

// Create a dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
});

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, processing, ready, error

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ pt: 4, pb: 4 }}>
        {status === 'error' || status !== 'ready' ? (
          <FileUpload
            setSessionId={setSessionId}
            setFileName={setFileName}
            setStatus={setStatus}
            status={status}
          />
        ) : (
          <ChatInterface
            sessionId={sessionId}
            fileName={fileName}
            status={status}
            setStatus={setStatus}
            resetSession={() => {
              setSessionId(null);
              setFileName(null);
              setStatus('idle');
            }}
          />
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App;

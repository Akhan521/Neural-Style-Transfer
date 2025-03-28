import { useState } from 'react'
import {
  Box, Typography, Button, TextField, Stack,
  CircularProgress, Grid, Card, CardContent,
  CardMedia, Fade,
} from '@mui/material'
import { motion } from 'framer-motion'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import axios from 'axios'
import './App.css'

// A custom motion box component for animations.
const MotionBox = motion(Box)

function App() {
  const [contentImage, setContentImage] = useState(null);
  const [styleImage, setStyleImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {

    e.preventDefault();

    // If either of our images are empty, we can't submit.
    if (!contentImage || !styleImage) {
      alert('Please select both content and style images.');
      return;
    }

    // Set loading to true to show our loading spinner.
    setLoading(true);

    // Create a new FormData object to hold our images.
    const formData = new FormData();
    formData.append('content', contentImage);
    formData.append('style', styleImage);

    // Send a POST request to our Flask backend with the images.
    try {
      const response = await axios.post('http://localhost:5000/neural-style-transfer', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        }
      });
      // Set our results to the response data.
      setResults(response.data);
    } catch (error) {
      console.error('Error uploading images:', error);
      alert('Error uploading images. Please try again.');
    } finally {
      // Set loading to false to hide our loading spinner.
      setLoading(false);
    }
  };

  return (
    <Box sx={{ padding: 4, maxWidth: 1200, mx: 'auto' }}>
      {/* Our title */}
      <Fade in timeout={1000}>
        <Typography variant="h3" align="center" color='primary' gutterBottom>
          Neural Style Transfer
        </Typography>
      </Fade>

      {/* Our form for uploading content and style images */}
      <Stack component="form" spacing={3} onSubmit={handleSubmit} alignItems="center" sx={{ mb: 4 }}>
        <TextField
          type="file"
          label="Content Image"
          slotProps={{
            inputLabel: {
              shrink: true,
            },
            inputProps: {
              accept: 'image/*',
            },
          }}
          onChange={(e) => setContentImage(e.target.files[0])}
          required
          fullWidth
        />
        <TextField
          type="file"
          label="Style Image"
          slotProps={{
            inputLabel: {
              shrink: true,
            },
            inputProps: {
              accept: 'image/*',
            },
          }}
          onChange={(e) => setStyleImage(e.target.files[0])}
          required
          fullWidth
        />
        <Button
          type='submit'
          variant='contained'
          color='primary'
          size='large'
          startIcon={<CloudUploadIcon />}
          disabled={loading}
          sx={{ py: 1.5 }}
        >
          {loading ? 'Processing...' : 'Perfom Neural Style Transfer'}
        </Button>
      </Stack>
          
      {/* Loading spinner */}
      {loading && (
        <MotionBox
          display="flex"
          flexDirection="column"
          alignItems="center"
          mt={4}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <CircularProgress size={60} color="primary" />
          <Typography variant="h6" mt={2}>
            Performing Neural Style Transfer between images...
          </Typography>
        </MotionBox>
      )}


    </Box>
  )
}

export default App

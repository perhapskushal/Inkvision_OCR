# Deploying Flask OCR App to Render

This guide will help you deploy your Flask OCR application to Render.com.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be in a GitHub repository

## Quick Deployment Steps

### 1. Push to GitHub

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit for Render deployment"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

### 2. Deploy on Render

1. Go to [render.com](https://render.com)
2. Click "New Web Service"
3. Connect your GitHub account
4. Select your repository
5. Render will automatically detect the `render.yaml` configuration
6. Click "Create Web Service"

### 3. Wait for Deployment

Render will:
- Install Tesseract OCR and language data
- Install Python dependencies
- Start your Flask application
- Provide you with a public URL

## Configuration

The `render.yaml` file automatically:
- Installs Tesseract OCR with English and Hindi language support
- Sets up your Flask environment
- Configures health checks

## Testing

Once deployed:
1. Visit your Render URL
2. Test the OCR functionality
3. Check the `/health` endpoint

## Custom Domain (Optional)

After deployment:
1. Go to your Render service settings
2. Add a custom domain
3. Configure DNS records

## Cost

- **Free Tier**: Available for testing
- **Paid Plans**: Start at $7/month for production use

## Support

- Render Documentation: [docs.render.com](https://docs.render.com)
- Render Community: [community.render.com](https://community.render.com) 
# Deploying Flask OCR App to Railway (with Tesseract)

This guide will help you deploy your Flask OCR application to Railway while keeping your Tesseract functionality.

## Why Railway Instead of Vercel?

- ✅ **Supports Tesseract**: Can install system packages including Tesseract OCR
- ✅ **No API Keys Needed**: Uses your existing Tesseract setup
- ✅ **Full Control**: More flexible environment
- ✅ **Free Tier**: Available for testing

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Git Repository**: Your code should be in a Git repository
3. **GitHub Account**: For easy deployment

## Deployment Steps

### 1. Prepare Your Code

1. **Use the Railway-compatible files**:
   - `app_railway.py` (instead of `app.py`)
   - `railway.json` (Railway configuration)
   - `nixpacks.toml` (Tesseract installation config)

2. **Rename files for deployment**:
   ```bash
   mv app_railway.py app.py
   ```

### 2. Deploy to Railway

#### Method 1: GitHub Integration (Recommended)
1. Push your code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will automatically detect it's a Python app

#### Method 2: Railway CLI
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login to Railway
railway login

# Deploy
railway up
```

### 3. Configure Environment (Optional)

Railway will automatically:
- Install Tesseract and language data
- Install Python dependencies
- Start your Flask app

## What Railway Does Automatically

### Tesseract Installation
Railway uses Nixpacks to install:
- `tesseract` - The OCR engine
- `tesseract-data-eng` - English language data
- `tesseract-data-hin` - Hindi language data

### Path Configuration
The app automatically uses the correct Tesseract path:
```python
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

### Language Support
Railway installs both English and Hindi language data, so your Nepali OCR will work.

## Testing Your Deployment

1. **Health Check**: Visit your Railway URL + `/health`
2. **Home Page**: Visit your Railway URL
3. **OCR Function**: Test image upload and processing

## Custom Domain (Optional)

After deployment:
1. Go to your Railway project
2. Click "Settings"
3. Add custom domain

## Cost

- **Free Tier**: $5 credit/month (enough for testing)
- **Paid Plans**: Start at $5/month for more usage

## Troubleshooting

### Common Issues

1. **Tesseract Not Found**: Check that `nixpacks.toml` is in your repo
2. **Language Data Missing**: Railway should install it automatically
3. **Build Failures**: Check Railway logs for Python dependency issues

### Debugging

- Check Railway deployment logs
- Use the `/health` endpoint to verify Tesseract is working
- Test with simple images first

## Advantages Over Vercel

| Feature | Railway | Vercel |
|---------|---------|--------|
| Tesseract Support | ✅ Yes | ❌ No |
| System Packages | ✅ Yes | ❌ No |
| File Storage | ✅ Persistent | ❌ Temporary |
| Function Timeout | ✅ 5 minutes | ❌ 30 seconds |
| Custom Domains | ✅ Yes | ✅ Yes |
| Free Tier | ✅ $5 credit | ✅ 100GB bandwidth |

## Migration from Vercel

If you've already deployed to Vercel:

1. **Keep both deployments**: Vercel for simple OCR, Railway for full Tesseract
2. **Use Railway for production**: Better for your use case
3. **Update your domain**: Point your custom domain to Railway

## Next Steps

1. Deploy to Railway
2. Test OCR functionality
3. Set up custom domain (optional)
4. Monitor usage and costs

## Support

- Railway Documentation: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- GitHub Issues: For code-specific problems 
# Deploying Flask OCR App to Render

This guide will help you deploy your Flask OCR application to Render with full Tesseract support.

## Why Render?

- ✅ **Full Tesseract Support**: Can install system packages including Tesseract OCR
- ✅ **No API Keys Needed**: Uses your existing Tesseract setup
- ✅ **Free Tier Available**: Perfect for testing and small projects
- ✅ **Easy Deployment**: Simple GitHub integration
- ✅ **Custom Domains**: Free custom domains included

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **GitHub Account**: For easy deployment

## Quick Deployment Steps

### 1. Prepare Your Code

1. **Rename the Render-compatible file**:
   ```bash
   mv app_render.py app.py
   mv requirements_render.txt requirements.txt
   ```

2. **Make sure these files are in your repository**:
   - `app.py` (your Flask app)
   - `requirements.txt` (Python dependencies)
   - `render.yaml` (Render configuration)
   - `build.sh` (Tesseract installation script)
   - `templates/` (your HTML templates)
   - `fonts/` (your fonts)
   - `models/` (your trained models)

### 2. Deploy to Render

1. **Go to Render Dashboard**:
   - Visit [render.com](https://render.com)
   - Sign up/Login with your GitHub account

2. **Create New Web Service**:
   - Click "New +"
   - Select "Web Service"
   - Connect your GitHub repository

3. **Configure the Service**:
   - **Name**: `flask-ocr-app` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (for testing)

4. **Deploy**:
   - Click "Create Web Service"
   - Render will automatically build and deploy your app

## What Render Does Automatically

### System Dependencies Installation
The `build.sh` script installs:
- `tesseract-ocr` - The OCR engine
- `tesseract-ocr-eng` - English language data
- `tesseract-ocr-hin` - Hindi language data (for Nepali support)

### Python Dependencies
Render installs all packages from `requirements.txt`:
- Flask and web framework
- OpenCV for image processing
- PyTesseract for OCR
- Gunicorn for production server

### Environment Configuration
- Automatically sets `PORT` environment variable
- Uses Linux environment (compatible with your Tesseract path)

## Testing Your Deployment

1. **Health Check**: Visit `https://your-app-name.onrender.com/health`
2. **Home Page**: Visit `https://your-app-name.onrender.com/`
3. **OCR Function**: Test image upload and processing

## Custom Domain (Free)

After deployment:
1. Go to your Render service dashboard
2. Click "Settings"
3. Scroll to "Custom Domains"
4. Add your domain (e.g., `ocr.yourdomain.com`)
5. Update your DNS records as instructed

## Cost

- **Free Tier**: 
  - 750 hours/month (enough for always-on service)
  - 512 MB RAM
  - Shared CPU
  - Perfect for testing and small projects

- **Paid Plans**: Start at $7/month for more resources

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check Render build logs
   - Ensure `build.sh` is executable: `chmod +x build.sh`
   - Verify all files are in your repository

2. **Tesseract Not Found**:
   - Check that `build.sh` is in your repository
   - Verify the Tesseract path in `app.py`: `/usr/bin/tesseract`

3. **Import Errors**:
   - Check that all dependencies are in `requirements.txt`
   - Some packages might need system dependencies

### Debugging

- **Build Logs**: Check Render dashboard for build errors
- **Runtime Logs**: View logs in the Render dashboard
- **Health Check**: Use `/health` endpoint to verify deployment

## Performance Optimization

### For Production Use

1. **Upgrade Plan**: Consider paid plan for better performance
2. **Image Optimization**: Compress images before upload
3. **Caching**: Implement caching for processed results
4. **CDN**: Use Render's built-in CDN for static files

### Memory Management

- Free tier has 512 MB RAM limit
- Large images might cause memory issues
- Consider resizing images before processing

## Environment Variables

You can add environment variables in Render dashboard:
- `FLASK_ENV`: Set to `production`
- `DEBUG`: Set to `False`
- Custom variables for your app

## Monitoring

Render provides:
- **Uptime Monitoring**: Automatic health checks
- **Logs**: Real-time application logs
- **Metrics**: Performance metrics
- **Alerts**: Email notifications for downtime

## Advantages Over Other Platforms

| Feature | Render | Vercel | Railway | Heroku |
|---------|--------|--------|---------|--------|
| Tesseract Support | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| Free Tier | ✅ 750h/month | ✅ 100GB | ✅ $5 credit | ❌ Credit card |
| Custom Domains | ✅ Free | ✅ Yes | ✅ Yes | ✅ Paid |
| Build Time | ✅ 10 min | ✅ 5 min | ✅ 5 min | ✅ 15 min |
| Ease of Use | ✅ Very Easy | ✅ Easy | ✅ Easy | ✅ Medium |

## Next Steps

1. **Deploy to Render** using the steps above
2. **Test all functionality** (OCR, PDF generation, downloads)
3. **Set up custom domain** (optional)
4. **Monitor performance** and upgrade if needed
5. **Set up alerts** for downtime monitoring

## Support

- **Render Documentation**: [docs.render.com](https://docs.render.com)
- **Render Community**: [community.render.com](https://community.render.com)
- **GitHub Issues**: For code-specific problems

## Migration from Other Platforms

If you're migrating from another platform:

1. **Keep your existing deployment** until Render is working
2. **Test thoroughly** on Render
3. **Update DNS** to point to Render
4. **Monitor** for any issues
5. **Shut down** old deployment once confirmed working 
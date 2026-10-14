import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';
import { NestExpressApplication } from '@nestjs/platform-express';
import * as dotenv from 'dotenv';
import { join } from 'path';
import { HttpExceptionFilter } from './http-exception.filter';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { proxyConfig } from './proxy.config';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';

const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // Max requests per minute
});

async function bootstrap() {
  dotenv.config();

  const app = await NestFactory.create<NestExpressApplication>(AppModule);
  app.enableCors();
  app.use(limiter);
  app.use(helmet());
  app.useGlobalPipes(new ValidationPipe());
  app.useGlobalFilters(new HttpExceptionFilter());
  app.useStaticAssets(join(__dirname, '..', 'public'));
  app.setBaseViewsDir(join(__dirname, '..', 'views'));
  app.setViewEngine('hbs');

   Object.entries(proxyConfig).forEach(([route, config]) => {
    app.use(route, createProxyMiddleware(config));
  }); 

  const port = process.env.PORT || 5000;
  await app.listen(port, () => {
    console.log(`API Gateway is running on port ${port}`);
  });
}

bootstrap();

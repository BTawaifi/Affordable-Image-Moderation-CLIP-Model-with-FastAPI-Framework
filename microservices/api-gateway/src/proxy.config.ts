export const proxyConfig = {
  '/auth/*': {
    target: process.env.AUTH_SERVICE_URL || 'http://localhost:5010',
    changeOrigin: true,
  },
  '/images/*': {
    target: process.env.IMAGE_SERVICE_URL || 'http://localhost:5020',
    changeOrigin: true,
  },
  '/retrieve/*': {
    target: process.env.IMAGE_RETRIVAL_SERVICE_URL || 'http://localhost:5040',
    changeOrigin: true,
  },
};




/*
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';
import { NestExpressApplication } from '@nestjs/platform-express';
import * as dotenv from 'dotenv';
import { join } from 'path';
import { HttpExceptionFilter } from './http-exception.filter';
import * as proxy from 'http-proxy-middleware';
import { proxyConfig } from './proxy.config';

async function bootstrap() {
  dotenv.config();

  const app = await NestFactory.create<NestExpressApplication>(AppModule);
  app.enableCors();
  app.useGlobalPipes(new ValidationPipe());
  app.useGlobalFilters(new HttpExceptionFilter());
  app.useStaticAssets(join(__dirname, '..', 'public'));
  app.setBaseViewsDir(join(__dirname, '..', 'views'));
  app.setViewEngine('hbs');

  Object.entries(proxyConfig).forEach(([route, config]) => {
    app.use(route, proxy(config));
  });

  const port = process.env.PORT || 5050;
  await app.listen(port, () => {
    console.log(`API Gateway is running on port ${port}`);
  });
}

bootstrap();



export const proxyConfig = {
  '/clip/*': {
    target: 'http://clip-loadbalancer',
    changeOrigin: true,
  },
  '/resnet/*': {
    target: 'http://resnet-loadbalancer',
    changeOrigin: true,
  },
  '/images/*': {
    target: 'http://images-loadbalancer',
    changeOrigin: true,
  },
  '/flagging/*': {
    target: 'http://flagging-loadbalancer',
    changeOrigin: true,
  },
  '/analytics/*': {
    target: 'http://analytics-loadbalancer',
    changeOrigin: true,
  },
  '/auth/*': {
    target: 'http://auth-loadbalancer',
    changeOrigin: true,
  },
};

*/

export const proxyConfig = {
  '/clip/*': {
    target: 'http://localhost:3000',
    changeOrigin: true,
  },
  '/resnet/*': {
    target: 'http://localhost:4000',
    changeOrigin: true,
  },
  '/images/*': {
    target: 'http://localhost:5000',
    changeOrigin: true,
  },
  '/flagging/*': {
    target: 'http://localhost:6000',
    changeOrigin: true,
  },
  '/analytics/*': {
    target: 'http://localhost:7000',
    changeOrigin: true,
  },
  '/auth/*': {
    target: 'http://localhost:8000',
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

  const port = process.env.PORT || 8080;
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

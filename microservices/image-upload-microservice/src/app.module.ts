import { Module } from '@nestjs/common';
import { MulterModule } from '@nestjs/platform-express';
import { multerConfig } from './multer.config';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { ClipService } from './services/clip.service';
import { HttpModule } from '@nestjs/axios';
import { ResnetService } from './services/resnet.service';
import { APP_FILTER } from '@nestjs/core';
import { GlobalExceptionFilter } from './filters/global-exception.filter';
import { ImageUploadExceptionFilter } from './filters/image-upload-exception.filter';
import { ImageClassificationExceptionFilter } from './filters/image-classification-exception.filter';
import { FlaggingMicroserviceExceptionFilter } from './filters/flagging-microservice-exception.filter';
import { ImageUploadController } from './image-upload/image-upload.controller';

@Module({
  imports: [
    HttpModule,
    MulterModule.register(multerConfig), // causing some problems
    /* ClientsModule.register([
      {
        name: 'AUTH_MICROSERVICE',
        transport: Transport.TCP,
        options: {
          host: process.env.AUTH_MICROSERVICE_HOST || 'localhost',
          port: Number(process.env.AUTH_MICROSERVICE_PORT) || 8000,
        },
      },
    ]), */
  ],
  controllers: [ImageUploadController],
  providers: [
    ClipService,
    ResnetService,
    {
      provide: APP_FILTER,
      useClass: GlobalExceptionFilter,
    },
    {
      provide: APP_FILTER,
      useClass: ImageUploadExceptionFilter,
    },
    {
      provide: APP_FILTER,
      useClass: ImageClassificationExceptionFilter,
    },
    {
      provide: APP_FILTER,
      useClass: FlaggingMicroserviceExceptionFilter,
    },
  ],
})
export class AppModule { }

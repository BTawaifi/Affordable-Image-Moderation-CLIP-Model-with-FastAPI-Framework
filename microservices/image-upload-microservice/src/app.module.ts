import { Module } from '@nestjs/common';
import { AMQPQueuingService } from './services/amqp-queuing.service';
import { HttpModule } from '@nestjs/axios';
import { APP_FILTER } from '@nestjs/core';
import { GlobalExceptionFilter } from './filters/global-exception.filter';
import { ImageUploadExceptionFilter } from './filters/image-upload-exception.filter';
import { ImageClassificationExceptionFilter } from './filters/image-classification-exception.filter';
import { ImageUploadController } from './image-upload/image-upload.controller';
import { UploadToFileServerService } from './services/upload-to-fileServer.service';

@Module({
  imports: [
    HttpModule,
    /* ClientsModule.register([
      {
        name: 'AUTH_MICROSERVICE',
        transport: Transport.TCP,
        options: {
          host: process.env.AUTH_MICROSERVICE_HOST || 'localhost',
          port: Number(process.env.AUTH_MICROSERVICE_PORT) || 5010,
        },
      },
    ]), */
  ],
  controllers: [ImageUploadController],
  providers: [
    AMQPQueuingService,
    UploadToFileServerService,
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
  ],
})
export class AppModule {}

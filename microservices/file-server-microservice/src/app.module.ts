import { Module } from '@nestjs/common';
import { FileServerModule } from './file-server/file-server.module';
import { APP_FILTER } from '@nestjs/core';
import { ImageSavingExceptionFilter } from './filters/image-saving-exception.filter';
import { MulterModule } from '@nestjs/platform-express';
import { multerConfig } from './file-server/multer.config';

@Module({
  imports: [
    FileServerModule,
    MulterModule.register(multerConfig), // causing some problems
  ],
  controllers: [],
  providers: [
    {
      provide: APP_FILTER,
      useClass: ImageSavingExceptionFilter,
    },
  ],
})
export class AppModule { }

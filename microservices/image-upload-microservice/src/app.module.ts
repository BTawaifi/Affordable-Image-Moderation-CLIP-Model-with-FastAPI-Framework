import { Module } from '@nestjs/common';
import { MulterModule } from '@nestjs/platform-express';
import { ImagesController } from './images/images.controller';
import { multerConfig } from './multer.config';
import { ClientsModule, Transport } from '@nestjs/microservices';

@Module({
  imports: [
    MulterModule.register(multerConfig),// causing some problems
    ClientsModule.register([
      {
        name: 'AUTH_MICROSERVICE',
        transport: Transport.TCP,
        options: {
          host: process.env.AUTH_MICROSERVICE_HOST || 'localhost',
          port: Number(process.env.AUTH_MICROSERVICE_PORT) || 8000,
        },
      },
    ]),
  ],
  controllers: [ImagesController],
})
export class AppModule {}

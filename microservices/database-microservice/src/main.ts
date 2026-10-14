import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { MicroserviceOptions, Transport } from '@nestjs/microservices';
import { ConfigService } from '@nestjs/config';

async function bootstrap() {
  const app = await NestFactory.createMicroservice<MicroserviceOptions>(
    AppModule,
    {
      transport: Transport.RMQ,
      options: {
        urls: [process.env.RABBITMQ_URL || 'amqp://localhost:5672'],
        queue: 'classification_result_queue',
        queueOptions: { durable: true },
      },
    },
  );

  const configService = app.get(ConfigService);
  const port = configService.get<number>('PORT') || 5030;

  await app.listen();
  console.log(`Microservice is listening on port ${port}`);
}

bootstrap();

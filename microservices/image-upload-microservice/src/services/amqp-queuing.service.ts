import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ChannelWrapper, connect } from 'amqp-connection-manager';

@Injectable()
export class AMQPQueuingService {
  private readonly logger = new Logger(AMQPQueuingService.name);
  private amqpConnection: ChannelWrapper;

  constructor(private httpService: HttpService) {
    this.connectToAmqp();
  }

  private async connectToAmqp(): Promise<void> {
    try {
      const connection = await connect('amqp://localhost:5672'); // Replace with your RabbitMQ connection URI
      this.amqpConnection = connection.createChannel({
        json: true,
        setup: channel => {
          return Promise.all([
            //channel.assertExchange('imagesToBeProcessed', 'direct', { durable: true }),
            channel.assertQueue('classification', { durable: true }), //Queue needs changing if we are using other models than CLIP
            //channel.bindQueue('classification', 'imagesToBeProcessed', 'classification'),
          ]);
        },
      });
    } catch (error) {
      this.logger.error(`Error connecting to RabbitMQ: ${(error as Error).message}`);
    }
  }

  async send(classes: string, location: string, classifier = 'CLIP', taskId: string, request: Request) {
    try {
      const payload = {
        classes,
        location,
        classifier,
        taskId,
        userId: request.headers['userId'],
      };

      await this.amqpConnection.sendToQueue('classification', payload); //concider direct exchange for different models

      // Log the message publishing success
      this.logger.log('Image classification message published successfully');
    } catch (error) {
      this.logger.error(`Error classifying image: ${(error as Error).message}`);
      throw new Error('Failed to classify image');
    }
  }
}

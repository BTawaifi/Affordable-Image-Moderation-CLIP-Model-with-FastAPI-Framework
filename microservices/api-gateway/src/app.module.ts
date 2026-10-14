import { Module } from '@nestjs/common';
import { APP_FILTER } from '@nestjs/core';
import { HttpExceptionFilter } from './http-exception.filter';
import { ProxyModule } from './proxy/proxy.module';
import { ClientsModule, Transport } from '@nestjs/microservices';
@Module({
  imports: [
    ProxyModule,
    /* ClientsModule.register([
      {
        name: 'AUTH_MICROSERVICE',
        transport: Transport.TCP,
        options: {
          port: 5010, // Update with the Auth microservice port
        },
      },
    ]), */
  ], // Import the ProxyModule here
  providers: [
    {
      provide: APP_FILTER,
      useClass: HttpExceptionFilter,
    },
  ],
})
export class AppModule {}

import { Module } from '@nestjs/common';
import { APP_FILTER } from '@nestjs/core';
import { HttpExceptionFilter } from './http-exception.filter';
import { ProxyModule } from './proxy/proxy.module';
import { ProxyController } from './proxy/proxy.controller';

@Module({
  imports: [ProxyModule],
  providers: [
    {
      provide: APP_FILTER,
      useClass: HttpExceptionFilter,
    },
  ],
  controllers: [ProxyController],
})
export class AppModule {}

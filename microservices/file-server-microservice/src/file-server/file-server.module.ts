import { Module } from '@nestjs/common';
import { FileServerController } from './file-server.controller';

@Module({
  providers: [],
  controllers: [FileServerController],
})
export class FileServerModule {}

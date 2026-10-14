import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ImageRetrievalModule } from './image-retrieval/image-retrieval.module';
import { ConfigModule, ConfigService } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule.forRoot(),
    TypeOrmModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        type: 'mysql',
        host: configService.get<string>('DATABASE_HOST', 'localhost'),
        port: configService.get<number>('DATABASE_PORT', 3306),
        username: configService.get<string>('DATABASE_USERNAME', 'root'),
        password: configService.get<string>('DATABASE_PASSWORD', 'root'),
        database: configService.get<string>(
          'DATABASE_NAME',
          'image_moderation_classification_result',
        ),
        entities: [__dirname + '/**/*.entity{.ts,.js}'],
        synchronize: true,
      }),
      inject: [ConfigService],
    }),
    ImageRetrievalModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}

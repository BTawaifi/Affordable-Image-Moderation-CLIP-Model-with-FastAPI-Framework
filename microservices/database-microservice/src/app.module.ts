import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ModerationResultModule } from './moderation-result/moderation-result.module';
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
    ModerationResultModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}

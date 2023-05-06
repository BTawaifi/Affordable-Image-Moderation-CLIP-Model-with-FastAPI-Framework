// app.module.ts
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AuthModule } from './auth/auth.module';
import { UserModule } from './entity/user/user.module';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: 'localhost',
      port: 5432,
      username: 'your-username',
      password: '***REDACTED***',
      database: 'your-database',
      entities: [__dirname + '/**/*.entity{.ts,.js}'], // Use a glob pattern to load all entities
      synchronize: true,
    }),
    UserModule,
    AuthModule,
  ],
})
export class AppModule {}

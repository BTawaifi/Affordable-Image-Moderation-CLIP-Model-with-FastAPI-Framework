// app.module.ts
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AuthModule } from './auth/auth.module';
import { UserModule } from './entity/user/user.module';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'mysql', // Change the type to 'mysql'
      host: 'localhost', // Update the host if needed
      port: 3306, // Update the port if needed
      username: 'root',
      password: '***REDACTED***',
      database: 'image_moderation_auth', // Use the name of the schema you created
      entities: [__dirname + '/**/*.entity{.ts,.js}'], // Use a glob pattern to load all entities
      synchronize: true,
    }),
    UserModule,
    AuthModule,
  ],
})
export class AppModule {}

import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ClassificationResult } from './entities/classification-results.entity';
import { LabelProbability } from './entities/label-probabilities.entity';
import { ModerationResultController } from './moderation-result.controller';
import { ModerationResultService } from './moderation-result.service';

@Module({
  imports: [TypeOrmModule.forFeature([ClassificationResult, LabelProbability])],
  controllers: [ModerationResultController],
  providers: [ModerationResultService],
})
export class ModerationResultModule { }

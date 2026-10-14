import { Controller, Get, Param, UseInterceptors } from '@nestjs/common';
import { ImageRetrievalService } from './image-retrieval.service';
import { ClassificationResult } from './entities/classification-results.entity';
import { ImageRetrievalInterceptor } from './interceptors/image-retrieval.interceptor';

@Controller('retrieve')
export class ImageRetrievalController {
  constructor(private readonly imageRetrievalService: ImageRetrievalService) {}

  @UseInterceptors(ImageRetrievalInterceptor)
  @Get(':taskId')
  async getImageByTaskId(
    @Param('taskId') taskId: string,
  ): Promise<ClassificationResult> {
    return this.imageRetrievalService.getImageByTaskId(taskId);
  }
}
